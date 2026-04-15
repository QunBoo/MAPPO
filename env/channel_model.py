"""
Channel Model for AMAPPO satellite edge computing project.

Implements physical-layer channel models for all communication links in the
4-tier IoTD -> UAV -> LEO satellite -> Cloud architecture:

  Link type              | Model                   | Bandwidth
  -----------------------|-------------------------|----------
  G2U / U2G              | Rician fading           | 20 MHz
  G2S / S2G / U2S / S2U | Shadowed-Rician fading  | 15 MHz
  ISL / S2C              | Free-space path loss    | 1 GHz

Channel rate formula (Shannon capacity):
    R = B * log2(1 + P * |h|^2 / (N_0 * B))

where N_0 = -174 dBm/Hz (thermal noise spectral density).

Satellite links (U2S, ISL, S2C) include a combined Tx+Rx directional antenna
system gain G_sys that accounts for the highly directive antennas used on LEO
satellites and ground/UAV terminals.  Without this gain, free-space losses at
26 GHz over hundreds of kilometres make the SNR negative and rates vanish —
contrary to the Gbps/tens-of-Mbps figures expected in the design specification.

Default antenna system gains (physically reasonable for 26 GHz satellite links):
    G_SAT_SYSTEM  = 10^5  (50 dB) — for UAV/Ground ↔ Satellite links
    G_ISL_SYSTEM  = 10^6  (60 dB) — for ISL and Satellite-to-Cloud links

All positions are in metres (x, y, z).
All rates are returned in bits/second.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_SPEED_OF_LIGHT: float = 3e8           # m/s
_BOLTZMANN_DBM_HZ: float = -174.0      # dBm/Hz  (thermal noise spectral density)

# Carrier frequencies
_FC_G2U: float = 2.4e9                 # Hz  — Rician (ground-to-UAV)
_FC_SAT: float = 26e9                  # Hz  — satellite links (Ka-band)

# ---------------------------------------------------------------------------
# Bandwidth presets (Hz)
# ---------------------------------------------------------------------------
BW_G2U: float = 20e6                   # 20 MHz  — G2U / U2G
BW_SAT: float = 15e6                   # 15 MHz  — G2S / S2G / U2S / S2U
BW_ISL: float = 1e9                    # 1 GHz   — ISL / S2C

# ---------------------------------------------------------------------------
# Transmit power presets (W)
# ---------------------------------------------------------------------------
P_IOTD_MAX: float = 1.0               # IoTD maximum transmit power
P_UAV: float = 2.0                    # UAV transmit power
P_SAT: float = 5.0                    # LEO satellite transmit power
P_CLOUD: float = 5.0                  # Cloud transmit power

# ---------------------------------------------------------------------------
# Rician fading parameters (G2U / U2G)
# ---------------------------------------------------------------------------
_K0: float = 1.0                      # K-factor at 0° elevation
_ETA_K: float = 10.0                  # elevation-angle scaling coefficient

# ---------------------------------------------------------------------------
# Shadowed-Rician parameters
# Table: (g, Omega) — shape parameter and mean power of LoS component
# Source: Lutz et al. / Abdi et al. land-mobile satellite channel model
# ---------------------------------------------------------------------------
_SHADOW_PARAMS: dict[str, Tuple[float, float]] = {
    "Light":   (19.4,  1.29),
    "Average": (10.1,  0.835),
    "Heavy":   (0.739, 8.97e-4),
}

# ---------------------------------------------------------------------------
# Effective antenna system gains for satellite links
# These account for the directional (dish/phased-array) antennas used on LEO
# satellites and ground/UAV terminals.  At 26 GHz, a 10–30 cm aperture dish
# achieves 26–36 dBi, so combined Tx+Rx gains of 50–60 dB are physically
# realistic.
# ---------------------------------------------------------------------------
G_SAT_SYSTEM: float = 1e5             # 50 dB — G2S / S2G / U2S / S2U links
G_ISL_SYSTEM: float = 1e6             # 60 dB — ISL / S2C links


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _noise_power_w(bandwidth_hz: float) -> float:
    """Return total thermal noise power in Watts: N0 * B."""
    n0_w_per_hz = 10.0 ** ((_BOLTZMANN_DBM_HZ - 30.0) / 10.0)  # dBm/Hz -> W/Hz
    return n0_w_per_hz * bandwidth_hz


def _shannon_rate(snr: float, bandwidth_hz: float) -> float:
    """Return Shannon capacity R = B * log2(1 + SNR) in bits/second."""
    return bandwidth_hz * math.log2(1.0 + max(snr, 0.0))


def _free_space_path_loss(distance_m: float, freq_hz: float) -> float:
    """
    Free-space path gain (dimensionless, ≤1).

    PL(d) = (lambda / (4*pi*d))^2,  lambda = c / f_c
    """
    if distance_m <= 0.0:
        return 1.0
    wavelength = _SPEED_OF_LIGHT / freq_hz
    return (wavelength / (4.0 * math.pi * distance_m)) ** 2


def _shadowed_rician_mean_gain(shadow_level: str) -> float:
    """
    Mean channel power gain E[|h|^2] for the Shadowed-Rician fading model.

    Closed-form approximation (Abdi et al.):
        E[|h|^2] = Omega * g / (g + 1) + 1 / (g + 1)

    Returns a dimensionless value (path loss applied separately).
    """
    if shadow_level not in _SHADOW_PARAMS:
        raise ValueError(
            f"shadow_level must be one of {list(_SHADOW_PARAMS.keys())}, "
            f"got '{shadow_level}'"
        )
    g, omega = _SHADOW_PARAMS[shadow_level]
    return omega * g / (g + 1.0) + 1.0 / (g + 1.0)


def _rician_k_factor(elevation_angle_rad: float) -> float:
    """
    Rician K-factor as a function of elevation angle (Bai & Heath model).

    K(theta) = K_0 * exp(eta_K * sin(theta))
    """
    return _K0 * math.exp(_ETA_K * math.sin(elevation_angle_rad))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ChannelModel:
    """
    Physical-layer channel models for the 4-tier satellite edge computing system.

    All rate methods return achievable transmission rate in **bits/second**,
    computed via the Shannon formula R = B * log2(1 + SNR).

    Satellite link methods (shadowed_rician_rate, free_space_rate, u2s_rate,
    isl_rate, s2c_rate) include an effective antenna system gain parameter
    ``antenna_gain`` (default: G_SAT_SYSTEM or G_ISL_SYSTEM) that models the
    combined transmit and receive directional antenna gain of the satellite
    communication system.
    """

    # ------------------------------------------------------------------
    # 1. Rician fading  (G2U / U2G)
    # ------------------------------------------------------------------

    @staticmethod
    def rician_rate(
        distance_3d: float,
        elevation_angle_rad: float,
        bandwidth_hz: float,
        tx_power_w: float,
        freq_hz: float = _FC_G2U,
    ) -> float:
        """
        Rician-fading channel rate for ground-to-UAV (G2U) and UAV-to-ground
        (U2G) links.

        The mean channel power gain is:
            E[|h|^2] = PL(d) = (lambda / (4*pi*d))^2

        This implementation returns ergodic capacity using mean path-loss SNR.
        The K-factor is computed but does not modulate the returned rate in this
        deterministic approximation.

        Parameters
        ----------
        distance_3d : float
            3-D Euclidean distance between transmitter and receiver [m].
        elevation_angle_rad : float
            Elevation angle from the ground node to the UAV [rad].
        bandwidth_hz : float
            Channel bandwidth [Hz].
        tx_power_w : float
            Transmit power [W].
        freq_hz : float
            Carrier frequency [Hz]. Default: 2.4 GHz.

        Returns
        -------
        float
            Achievable rate [bits/s].
        """
        pl = _free_space_path_loss(distance_3d, freq_hz)
        # K-factor characterises the Rician channel; retained for documentation/subclass use.
        # This method uses mean SNR (ergodic capacity), so K does not alter the scalar output.
        _k = _rician_k_factor(elevation_angle_rad)
        noise_w = _noise_power_w(bandwidth_hz)
        snr = tx_power_w * pl / noise_w
        return _shannon_rate(snr, bandwidth_hz)

    # ------------------------------------------------------------------
    # 2. Shadowed-Rician fading  (G2S / S2G / U2S / S2U)
    # ------------------------------------------------------------------

    @staticmethod
    def shadowed_rician_rate(
        distance: float,
        bandwidth_hz: float,
        tx_power_w: float,
        shadow_level: str = "Average",
        freq_hz: float = _FC_SAT,
        antenna_gain: float = G_SAT_SYSTEM,
    ) -> float:
        """
        Shadowed-Rician channel rate for ground/UAV ↔ satellite links.

        The effective channel power gain is:
            G_eff = PL(d) * E[|h|^2_SR] * antenna_gain

        where E[|h|^2_SR] = Omega*g/(g+1) + 1/(g+1) is the mean Shadowed-Rician
        fading gain and antenna_gain accounts for the directional antennas used
        on the satellite and UAV/ground terminals.

        Parameters
        ----------
        distance : float
            Distance between transmitter and receiver [m].
        bandwidth_hz : float
            Channel bandwidth [Hz].
        tx_power_w : float
            Transmit power [W].
        shadow_level : str
            Shadowing severity: 'Light', 'Average', or 'Heavy'.
        freq_hz : float
            Carrier frequency [Hz]. Default: 26 GHz.
        antenna_gain : float
            Combined Tx+Rx antenna system gain (linear). Default: G_SAT_SYSTEM.

        Returns
        -------
        float
            Achievable rate [bits/s].
        """
        pl = _free_space_path_loss(distance, freq_hz)
        sr_gain = _shadowed_rician_mean_gain(shadow_level)
        channel_gain = pl * sr_gain * antenna_gain
        noise_w = _noise_power_w(bandwidth_hz)
        snr = tx_power_w * channel_gain / noise_w
        return _shannon_rate(snr, bandwidth_hz)

    # ------------------------------------------------------------------
    # 3. Free-space path loss  (ISL / S2C)
    # ------------------------------------------------------------------

    @staticmethod
    def free_space_rate(
        distance: float,
        bandwidth_hz: float,
        tx_power_w: float,
        freq_hz: float = _FC_SAT,
        antenna_gain: float = G_ISL_SYSTEM,
    ) -> float:
        """
        Free-space-path-loss channel rate for inter-satellite (ISL) and
        satellite-to-cloud (S2C) links.

        SNR = P * PL(d) * antenna_gain / (N_0 * B)

        Parameters
        ----------
        distance : float
            Distance between transmitter and receiver [m].
        bandwidth_hz : float
            Channel bandwidth [Hz].
        tx_power_w : float
            Transmit power [W].
        freq_hz : float
            Carrier frequency [Hz]. Default: 26 GHz.
        antenna_gain : float
            Combined Tx+Rx antenna system gain (linear). Default: G_ISL_SYSTEM.

        Returns
        -------
        float
            Achievable rate [bits/s].
        """
        pl = _free_space_path_loss(distance, freq_hz)
        noise_w = _noise_power_w(bandwidth_hz)
        snr = tx_power_w * pl * antenna_gain / noise_w
        return _shannon_rate(snr, bandwidth_hz)

    # ------------------------------------------------------------------
    # 4. Convenience methods — specific links
    # ------------------------------------------------------------------

    @staticmethod
    def g2u_rate(
        uav_pos: Tuple[float, float, float],
        device_pos: Tuple[float, float, float],
        tx_power_w: float,
    ) -> float:
        """
        Ground-to-UAV (G2U) transmission rate using Rician fading.

        Parameters
        ----------
        uav_pos : (x, y, z) [m]
            3-D position of the UAV (z = altitude above ground).
        device_pos : (x, y, z) [m]
            3-D position of the IoT device (z is typically 0).
        tx_power_w : float
            IoTD transmit power [W]. Should be in [0, P_IOTD_MAX].

        Returns
        -------
        float
            Rate [bits/s].
        """
        dx = uav_pos[0] - device_pos[0]
        dy = uav_pos[1] - device_pos[1]
        dz = uav_pos[2] - device_pos[2]

        dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        dist_horiz = math.sqrt(dx**2 + dy**2)

        # Elevation angle: angle at the device looking up toward the UAV
        if dist_horiz < 1e-9:
            elev = math.pi / 2.0      # UAV directly overhead
        else:
            elev = math.atan2(abs(dz), dist_horiz)

        return ChannelModel.rician_rate(
            distance_3d=dist_3d,
            elevation_angle_rad=elev,
            bandwidth_hz=BW_G2U,
            tx_power_w=tx_power_w,
            freq_hz=_FC_G2U,
        )

    @staticmethod
    def u2s_rate(
        sat_pos: Tuple[float, float, float],
        uav_pos: Tuple[float, float, float],
        tx_power_w: float,
        shadow: str = "Average",
    ) -> float:
        """
        UAV-to-Satellite (U2S) transmission rate using Shadowed-Rician fading.

        Parameters
        ----------
        sat_pos : (x, y, z) [m]
            3-D position of the LEO satellite.
        uav_pos : (x, y, z) [m]
            3-D position of the UAV.
        tx_power_w : float
            UAV transmit power [W].
        shadow : str
            Shadowing level: 'Light', 'Average', or 'Heavy'.

        Returns
        -------
        float
            Rate [bits/s].
        """
        dx = sat_pos[0] - uav_pos[0]
        dy = sat_pos[1] - uav_pos[1]
        dz = sat_pos[2] - uav_pos[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        return ChannelModel.shadowed_rician_rate(
            distance=dist,
            bandwidth_hz=BW_SAT,
            tx_power_w=tx_power_w,
            shadow_level=shadow,
            freq_hz=_FC_SAT,
            antenna_gain=G_SAT_SYSTEM,
        )

    @staticmethod
    def isl_rate(dist_km: float, tx_power_w: float) -> float:
        """
        Inter-Satellite Link (ISL) transmission rate using free-space path loss.

        Note: distance parameter is in kilometres (unlike g2u_rate/u2s_rate
        which take positions in metres).

        Parameters
        ----------
        dist_km : float
            Distance between satellites [km].
        tx_power_w : float
            Transmit power [W].

        Returns
        -------
        float
            Rate [bits/s].
        """
        distance_m = dist_km * 1000  # convert km -> m
        return ChannelModel.free_space_rate(
            distance=distance_m,
            bandwidth_hz=BW_ISL,
            tx_power_w=tx_power_w,
            freq_hz=_FC_SAT,
            antenna_gain=G_ISL_SYSTEM,
        )

    @staticmethod
    def s2c_rate(dist_km: float, tx_power_w: float) -> float:
        """
        Satellite-to-Cloud (S2C) transmission rate using free-space path loss.

        Uses the same link model as ISL.  Provided as a separate method for
        semantic clarity.

        Note: distance parameter is in kilometres (unlike g2u_rate/u2s_rate
        which take positions in metres).

        Parameters
        ----------
        dist_km : float
            Distance from satellite to cloud gateway [km].
        tx_power_w : float
            Satellite transmit power [W].

        Returns
        -------
        float
            Rate [bits/s].
        """
        distance_m = dist_km * 1000  # convert km -> m
        return ChannelModel.free_space_rate(
            distance=distance_m,
            bandwidth_hz=BW_ISL,
            tx_power_w=tx_power_w,
            freq_hz=_FC_SAT,
            antenna_gain=G_ISL_SYSTEM,
        )


# ---------------------------------------------------------------------------
# Verification / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cm = ChannelModel()

    _SEP = "=" * 72

    print(_SEP)
    print("Channel Model Verification — AMAPPO 4-tier satellite edge computing")
    print(_SEP)

    # ------------------------------------------------------------------
    # Test 1: G2U Rician fading
    # UAV altitude = 50 m, varying horizontal distances
    # ------------------------------------------------------------------
    print("\n[1] G2U Rician fading  (B=20 MHz, f=2.4 GHz, P_IoTD=1 W, H_uav=50 m)")
    print(f"  {'H_dist (m)':>12}  {'3D dist (m)':>12}  {'Elev (deg)':>11}  {'Rate (Mbps)':>12}")
    print("  " + "-" * 54)

    uav_height = 50.0
    for h_dist in [0.1, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        uav_p = (0.0, 0.0, uav_height)
        dev_p = (h_dist, 0.0, 0.0)
        dist_3d = math.sqrt(h_dist**2 + uav_height**2)
        elev_deg = math.degrees(math.atan2(uav_height, h_dist))
        rate_mbps = cm.g2u_rate(uav_p, dev_p, tx_power_w=P_IOTD_MAX) / 1e6
        print(f"  {h_dist:>12.1f}  {dist_3d:>12.2f}  {elev_deg:>11.2f}  {rate_mbps:>12.3f}")

    # Specific assertion: at H=50m, h=100m the rate must exceed 1 Mbps
    r_check = cm.g2u_rate((0.0, 0.0, 50.0), (100.0, 0.0, 0.0), tx_power_w=1.0)
    assert r_check > 1e6, f"G2U rate too low: {r_check/1e6:.3f} Mbps"
    print(f"\n  Check (H=50m, h=100m): {r_check/1e6:.3f} Mbps  [expect several Mbps to ~100+ Mbps] OK")

    # ------------------------------------------------------------------
    # Test 2: Shadowed-Rician fading
    # U2S, distance = 600 km (typical LEO slant range)
    # ------------------------------------------------------------------
    print(f"\n[2] Shadowed-Rician fading  (B=15 MHz, f=26 GHz, P_UAV=2 W, dist=600 km)")
    print(f"  {'Shadow':>10}  {'g':>7}  {'Omega':>9}  {'SR gain':>10}  {'Rate (Mbps)':>12}")
    print("  " + "-" * 56)

    dist_u2s_m = 600e3
    rates_sr: dict[str, float] = {}
    for level in ("Light", "Average", "Heavy"):
        g, omega = _SHADOW_PARAMS[level]
        mean_g = _shadowed_rician_mean_gain(level)
        rate = cm.shadowed_rician_rate(
            distance=dist_u2s_m,
            bandwidth_hz=BW_SAT,
            tx_power_w=P_UAV,
            shadow_level=level,
        )
        rates_sr[level] = rate
        print(f"  {level:>10}  {g:>7.3f}  {omega:>9.4f}  {mean_g:>10.6f}  {rate/1e6:>12.4f}")

    assert rates_sr["Light"] > rates_sr["Average"] > rates_sr["Heavy"], (
        f"Shadow ordering violated: Light={rates_sr['Light']/1e6:.4f}, "
        f"Average={rates_sr['Average']/1e6:.4f}, Heavy={rates_sr['Heavy']/1e6:.4f}"
    )
    print("\n  Check: Light > Average > Heavy ordering  OK")

    # ------------------------------------------------------------------
    # Test 3: Free-space ISL / S2C
    # ------------------------------------------------------------------
    print(f"\n[3] Free-space ISL / S2C  (B=1 GHz, f=26 GHz, P=5 W, G_sys=60 dB)")
    print(f"  {'Dist (km)':>10}  {'PL (dB)':>10}  {'SNR (dB)':>10}  {'Rate (Gbps)':>12}")
    print("  " + "-" * 48)

    noise_isl = _noise_power_w(BW_ISL)
    for d_km in [100.0, 500.0, 1000.0, 5000.0]:
        pl = _free_space_path_loss(d_km * 1e3, _FC_SAT)
        pl_db = 10 * math.log10(pl)
        snr = P_SAT * pl * G_ISL_SYSTEM / noise_isl
        snr_db = 10 * math.log10(snr) if snr > 0 else float("-inf")
        rate = cm.isl_rate(d_km, tx_power_w=P_SAT)
        print(f"  {d_km:>10.0f}  {pl_db:>10.2f}  {snr_db:>10.2f}  {rate/1e9:>12.4f}")

    rate_1000km = cm.isl_rate(1000.0, tx_power_w=P_SAT)
    assert rate_1000km > 1e8, (
        f"ISL rate at 1000 km should be > 0.1 Gbps, got {rate_1000km/1e9:.4f} Gbps"
    )
    print(f"\n  Check (ISL 1000 km): {rate_1000km/1e9:.4f} Gbps  [expect ~Gbps range] OK")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print("Summary — representative link rates")
    print(_SEP)
    print(f"  {'Link':>24}  {'Config':>28}  {'Rate':>14}")
    print("  " + "-" * 72)

    rows = [
        ("G2U (Rician)",
         "H=50m, h=100m, P=1W",
         cm.g2u_rate((0.0, 0.0, 50.0), (100.0, 0.0, 0.0), 1.0)),
        ("G2U (Rician)",
         "H=50m, h=500m, P=1W",
         cm.g2u_rate((0.0, 0.0, 50.0), (500.0, 0.0, 0.0), 1.0)),
        ("U2S Light (Sh-Rician)",
         "d=600km, P=2W",
         cm.u2s_rate((0.0, 0.0, 600e3), (0.0, 0.0, 0.0), P_UAV, "Light")),
        ("U2S Average (Sh-Rician)",
         "d=600km, P=2W",
         cm.u2s_rate((0.0, 0.0, 600e3), (0.0, 0.0, 0.0), P_UAV, "Average")),
        ("U2S Heavy (Sh-Rician)",
         "d=600km, P=2W",
         cm.u2s_rate((0.0, 0.0, 600e3), (0.0, 0.0, 0.0), P_UAV, "Heavy")),
        ("ISL (free-space)",
         "d=1000km, P=5W",
         cm.isl_rate(1000.0, P_SAT)),
        ("ISL (free-space)",
         "d=500km, P=5W",
         cm.isl_rate(500.0, P_SAT)),
        ("S2C (free-space)",
         "d=1000km, P=5W",
         cm.s2c_rate(1000.0, P_CLOUD)),
    ]

    for link, cfg, rate in rows:
        if rate >= 1e9:
            r_str = f"{rate/1e9:.4f} Gbps"
        elif rate >= 1e6:
            r_str = f"{rate/1e6:.4f} Mbps"
        elif rate >= 1e3:
            r_str = f"{rate/1e3:.4f} kbps"
        else:
            r_str = f"{rate:.4f} bps"
        print(f"  {link:>24}  {cfg:>28}  {r_str:>14}")

    print(f"\n{_SEP}")
    print("All verification checks passed.")
    print(_SEP)
