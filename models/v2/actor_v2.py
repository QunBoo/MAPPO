import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Optional, Tuple

# GRU input dims: h_v_t(64) + L_us_agg(64) + server_agg(64) + a_prev(8) = 200
_GRU_INPUT_DIM = 200
_GRU_HIDDEN = 64
_NODE_EMB_DIM = 64
_CONTEXT_DIM = _GRU_HIDDEN + _NODE_EMB_DIM  # 128

# LEO: discrete(4) + continuous(3: z, P, f)
_LEO_DISC = 4
_LEO_CONT = 3
# UAV: discrete(4) + continuous(5: z, P, f_m, f_k, q)
_UAV_DISC = 4
_UAV_CONT = 5


class ActorV2(nn.Module):
    def __init__(self, agent_type: str = "LEO"):
        super().__init__()
        if agent_type not in ("LEO", "UAV"):
            raise ValueError(f"agent_type must be 'LEO' or 'UAV', got '{agent_type}'")

        self.agent_type = agent_type

        # GRU init projection (initialize hidden state from graph_enc)
        self.hidden_init = nn.Linear(_NODE_EMB_DIM, _GRU_HIDDEN)

        # GRU
        self.gru = nn.GRU(
            input_size=_GRU_INPUT_DIM,
            hidden_size=_GRU_HIDDEN,
            batch_first=True,
        )

        # Discrete action head
        n_disc = _LEO_DISC if agent_type == "LEO" else _UAV_DISC
        self.discrete_head = nn.Linear(_CONTEXT_DIM, n_disc)

        # Continuous action head (mean)
        n_cont = _LEO_CONT if agent_type == "LEO" else _UAV_CONT
        self.cont_mean_head = nn.Linear(_CONTEXT_DIM, n_cont)
        self.log_std = nn.Parameter(torch.zeros(n_cont))

        self.n_disc = n_disc
        self.n_cont = n_cont

    def init_hidden(self, graph_enc: torch.Tensor) -> torch.Tensor:
        """
        Initialize GRU hidden state from global graph encoding.
        Args:
            graph_enc: (64,)
        Returns:
            h: (1, 1, 64)
        """
        h = F.relu(self.hidden_init(graph_enc))  # (64,)
        return h.unsqueeze(0).unsqueeze(0)       # (1, 1, 64)

    def _attention(self, w_t: torch.Tensor, node_embs: torch.Tensor) -> torch.Tensor:
        """
        Dot-product attention.
        Args:
            w_t:        (64,) or (B, 64)
            node_embs:  (N, 64)
        Returns:
            context: (64,) or (B, 64)
        """
        unbatched = w_t.dim() == 1
        if unbatched:
            w_t = w_t.unsqueeze(0)  # (1, 64)

        # scores: (B, N)
        scores = w_t @ node_embs.T
        alpha = F.softmax(scores, dim=-1)          # (B, N)
        context = alpha @ node_embs                # (B, 64)

        if unbatched:
            context = context.squeeze(0)           # (64,)
        return context

    def forward(
        self,
        h_v_t:      torch.Tensor,                  # (64,)
        L_us_agg:   torch.Tensor,                  # (64,)
        server_agg: torch.Tensor,                  # (64,)
        a_prev:     torch.Tensor,                  # (8,)
        h_prev:     Optional[torch.Tensor],        # (1, 1, 64) or None
        node_embs:  torch.Tensor,                  # (N, 64)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step inference.
        Returns:
            action:   (n_disc + n_cont,)  = (7,) LEO or (9,) UAV
            log_prob: scalar
            h_next:   (1, 1, 64)
        """
        # --- GRU ---
        gru_in = torch.cat([h_v_t, L_us_agg, server_agg, a_prev], dim=-1)  # (200,)
        gru_in = gru_in.unsqueeze(0).unsqueeze(0)  # (1, 1, 200)
        gru_out, h_next = self.gru(gru_in, h_prev)  # gru_out: (1, 1, 64)
        w_t = gru_out.squeeze(0).squeeze(0)         # (64,)

        # --- Attention ---
        context = self._attention(w_t, node_embs)  # (64,)
        c_t = torch.cat([w_t, context], dim=-1)    # (128,)

        # --- Discrete head ---
        disc_logits = self.discrete_head(c_t)      # (n_disc,)
        dist_disc = Categorical(logits=disc_logits)
        disc_action = dist_disc.sample()           # scalar
        log_prob_disc = dist_disc.log_prob(disc_action)  # scalar

        # --- Continuous head ---
        cont_mean = torch.sigmoid(self.cont_mean_head(c_t))  # (n_cont,) in [0,1]
        dist_cont = Normal(cont_mean, self.log_std.exp())
        cont_action = dist_cont.sample()           # (n_cont,)
        log_prob_cont = dist_cont.log_prob(cont_action).sum()  # scalar

        log_prob = log_prob_disc + log_prob_cont

        # Action format: [disc_logits(n_disc), cont(n_cont)]
        action = torch.cat([disc_logits.detach(), cont_action.detach()], dim=-1)

        return action, log_prob, h_next

    def evaluate(
        self,
        h_v_t:          torch.Tensor,   # (B, 64)
        L_us_agg:       torch.Tensor,   # (B, 64)
        server_agg:     torch.Tensor,   # (B, 64)
        a_prev:         torch.Tensor,   # (B, 8)
        h_prev:         Optional[torch.Tensor],  # (1, B, 64) or None
        node_embs:      torch.Tensor,   # (N, 64)
        discrete_actions: torch.Tensor, # (B,) int
        continuous_actions: Optional[torch.Tensor] = None,  # (B, n_cont)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch evaluation for PPO update.
        Returns:
            log_probs:  (B,)
            entropies:  (B,)
            h_next:     (1, B, 64)
        """
        B = h_v_t.shape[0]
        gru_in = torch.cat([h_v_t, L_us_agg, server_agg, a_prev], dim=-1)  # (B, 200)
        gru_in = gru_in.unsqueeze(1)  # (B, 1, 200)

        gru_out, h_next = self.gru(gru_in, h_prev)  # gru_out: (B, 1, 64)
        w_t = gru_out.squeeze(1)                    # (B, 64)

        context = self._attention(w_t, node_embs)   # (B, 64)
        c_t = torch.cat([w_t, context], dim=-1)     # (B, 128)

        disc_logits = self.discrete_head(c_t)       # (B, n_disc)
        dist_disc = Categorical(logits=disc_logits)
        log_probs_disc = dist_disc.log_prob(discrete_actions.long())  # (B,)
        entropies_disc = dist_disc.entropy()                           # (B,)

        # Continuous action head
        cont_mean = torch.sigmoid(self.cont_mean_head(c_t))  # (B, n_cont)
        dist_cont = Normal(cont_mean, self.log_std.exp())
        if continuous_actions is not None:
            log_probs_cont = dist_cont.log_prob(continuous_actions).sum(dim=-1)  # (B,)
        else:
            log_probs_cont = torch.zeros(B, device=h_v_t.device)
        entropies_cont = dist_cont.entropy().sum(dim=-1)  # (B,)

        log_probs = log_probs_disc + log_probs_cont  # (B,)
        entropies = entropies_disc + entropies_cont   # (B,)

        return log_probs, entropies, h_next
