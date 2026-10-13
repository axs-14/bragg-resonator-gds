"""
ABCD-matrix simulation of an N-period ALTERNATING-IMPEDANCE quarter-wave
Bragg resonator/mirror built from a "low" impedance line (Zl) and a
"high" impedance line (Zh):

    H - L - H - L - H - ... - H      (N periods of L-H, capped with H
                                       on both ends for a symmetric stack)

This is the classic quarter-wave stack (same idea as an optical
alternating-index Bragg mirror): each section is a quarter-wavelength
long at the design frequency f0, and stacking many periods deepens and
widens the reflection stopband. With N=1 this reduces to the simple
3-section H-L-H mirror.

Workflow
--------
1. Geometry (w, s) for the Zl and Zh CPW (or microstrip) lines
      -> zc_and_epseff() -> Zc_L, eps_eff_L, Zc_H, eps_eff_H
2. Quarter-wave length of each line at the design frequency f0:
      L = c0 / (4 * f0 * sqrt(eps_eff))
3. Build the alternating H-L-H-L-...-H stack for N_PERIODS
4. line_ABCD() for every segment, at every frequency, then cascade
5. ABCD_to_S() -> S11, S21   (+ input_impedance())
6. Plot |S11|, |S21| (dB) and Zin vs frequency, with the design
   frequency and per-line quarter-wave frequencies marked for a
   sanity check.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

c0 = 2.99792458e8  # speed of light in vacuum, m/s


# ---------------------------------------------------------------------
# 0. Elliptic-integral ratio K(k)/K'(k), Hilberg's closed-form approx.
# ---------------------------------------------------------------------
def kk_ratio(k):
    k = np.clip(k, 1e-9, 1 - 1e-9)
    kp = np.sqrt(1 - k**2)
    ratio = np.empty_like(k, dtype=float)

    hi = k >= 0.707  # 0.707 <= k <= 1
    lo = ~hi         # 0 <= k < 0.707

    ratio[hi] = (1 / np.pi) * np.log(2 * (1 + np.sqrt(k[hi])) / (1 - np.sqrt(k[hi])))
    ratio[lo] = np.pi / np.log(2 * (1 + np.sqrt(kp[lo])) / (1 - np.sqrt(kp[lo])))
    return ratio  # this is K(k)/K'(k)


# ---------------------------------------------------------------------
# 1a. Coplanar waveguide (CPW): centre trace width w, gap to ground s
# ---------------------------------------------------------------------
def cpw_zc_epseff(w, s, eps_r, h_sub=None):
    """
    w, s, h_sub : metres  (trace width, gap-to-ground, substrate thickness)
    eps_r       : substrate relative permittivity

    Returns (Zc, eps_eff).
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    s = np.atleast_1d(np.asarray(s, dtype=float))

    k = w / (w + 2 * s)
    kke = kk_ratio(np.array([k[0]]))[0] if w.size == 1 else kk_ratio(k)

    if h_sub is None:
        # semi-infinite substrate (thick chip, no backside ground effect)
        eps_eff = np.full_like(k, (eps_r + 1) / 2.0)
    else:
        # finite-thickness substrate correction (grounded-backside CPW)
        k1 = np.sinh(np.pi * w / (4 * h_sub)) / np.sinh(np.pi * (w + 2 * s) / (4 * h_sub))
        kk1 = kk_ratio(np.array([k1[0]]))[0] if w.size == 1 else kk_ratio(k1)
        eps_eff = 1 + (eps_r - 1) / 2.0 * (kk1 / kke)

    Zc = 30 * np.pi / np.sqrt(eps_eff) / kke
    return (Zc.item(), eps_eff.item()) if Zc.size == 1 else (Zc, eps_eff)


# ---------------------------------------------------------------------
# 1b. Microstrip alternative (Hammerstad-Jensen): trace width w,
#     substrate height h.
# ---------------------------------------------------------------------
def microstrip_zc_epseff(w, h, eps_r):
    u = w / h
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / u) ** -0.5
    if u <= 1:
        Zc = 60 / np.sqrt(eps_eff) * np.log(8 / u + u / 4)
    else:
        Zc = 120 * np.pi / (np.sqrt(eps_eff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
    return Zc, eps_eff


# ---------------------------------------------------------------------
# 1c. Single entry point: geometry -> (Zc, eps_eff), for either
#     line type. This is the "calculate impedance from dimensions"
#     function.
# ---------------------------------------------------------------------
def zc_and_epseff(w, s, eps_r, h_sub=None, kind="cpw"):
    """
    kind = "cpw"        : w = trace width, s = gap-to-ground
    kind = "microstrip" : w = trace width, s = substrate height (h)
    """
    if kind == "cpw":
        return cpw_zc_epseff(w, s, eps_r, h_sub)
    elif kind == "microstrip":
        return microstrip_zc_epseff(w, s, eps_r)
    else:
        raise ValueError("kind must be 'cpw' or 'microstrip'")


# ---------------------------------------------------------------------
# 2. Single transmission-line segment -> 2x2 ABCD matrix, vs frequency
# ---------------------------------------------------------------------
def line_ABCD(freq, Zc, length, eps_eff, alpha=0.0):
    freq = np.asarray(freq, dtype=float)
    k = 2 * np.pi * freq * np.sqrt(eps_eff) / c0
    Gamma = alpha + 1j * k
    GL = Gamma * length

    A = np.cosh(GL)
    B = Zc * np.sinh(GL)
    C = np.sinh(GL) / Zc
    D = np.cosh(GL)

    ABCD = np.empty((freq.size, 2, 2), dtype=complex)
    ABCD[:, 0, 0] = A
    ABCD[:, 0, 1] = B
    ABCD[:, 1, 0] = C
    ABCD[:, 1, 1] = D
    return ABCD


# ---------------------------------------------------------------------
# 3. Cascade a list of ABCD matrices, in order
# ---------------------------------------------------------------------
def cascade(ABCD_list):
    total = ABCD_list[0].copy()
    for M in ABCD_list[1:]:
        total = np.einsum('nij,njk->nik', total, M)
    return total


# ---------------------------------------------------------------------
# 4. ABCD -> S-parameters, and input impedance
# ---------------------------------------------------------------------
def ABCD_to_S(ABCD, Z0=50.0):
    A, B, C, D = (ABCD[:, 0, 0], ABCD[:, 0, 1], ABCD[:, 1, 0], ABCD[:, 1, 1])
    denom = A + B / Z0 + C * Z0 + D
    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S21 = 2.0 / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S22 = (-A + B / Z0 - C * Z0 + D) / denom
    return S11, S12, S21, S22


def input_impedance(ABCD, Z0=50.0):
    A, B, C, D = (ABCD[:, 0, 0], ABCD[:, 0, 1], ABCD[:, 1, 0], ABCD[:, 1, 1])
    return (A * Z0 + B) / (C * Z0 + D)


# ---------------------------------------------------------------------
# 5. Build the alternating H-L-H-L-...-H stack
# ---------------------------------------------------------------------
def build_stack(N_periods, symmetric=True):
    """
    Returns an ordered list of section labels ('H' or 'L').

    N_periods = number of (L, H) periods.
    symmetric=True  -> H (L H)^N_periods    (odd # of sections, mirror-
                        symmetric, standard quarter-wave Bragg stack)
    symmetric=False -> (H L)^N_periods      (even # of sections)
    """
    if symmetric:
        stack = ['H']
        for _ in range(N_periods):
            stack += ['L', 'H']
    else:
        stack = []
        for _ in range(N_periods):
            stack += ['H', 'L']
    return stack


# =======================================================================
if __name__ == "__main__":

    # ==== EDIT THESE to your actual design =================================
    LINE_KIND = "cpw"     # "cpw" or "microstrip"
    eps_r = 11.9          # substrate relative permittivity (e.g. silicon)
    h_sub = 300e-6         # substrate thickness, m (None = semi-infinite)

    f0 = 6e9              # Bragg / design (centre) frequency, Hz
                           # each section is a quarter-wavelength at f0

    N_PERIODS = 3           # number of (L, H) periods -> stack has
                           # 2*N_PERIODS + 1 sections (H at both ends);
                           # N_PERIODS=2 -> 5 sections ("5 quarters")
    SYMMETRIC = True       # True: H(LH)^N,  False: (HL)^N

    # Low-impedance line (Zl): wide trace, narrow gap -> lower Zc
    w_L, s_L = 42e-6, 4e-6

    # High-impedance line (Zh): narrow trace, wide gap -> higher Zc
    w_H, s_H = 4e-6, 23e-6

    alpha = 0.0             # attenuation, Np/m (0 = lossless)
    Z0 = 50.0                # port impedance, ohm
    # =======================================================================

    # ---- geometry -> impedance / eps_eff for each line type -------------
    Zc_L, eps_eff_L = zc_and_epseff(w_L, s_L, eps_r, h_sub, kind=LINE_KIND)
    Zc_H, eps_eff_H = zc_and_epseff(w_H, s_H, eps_r, h_sub, kind=LINE_KIND)

    # ---- quarter-wave length of each line at f0 --------------------------
    L_L = c0 / (4 * f0 * np.sqrt(eps_eff_L))
    L_H = c0 / (4 * f0 * np.sqrt(eps_eff_H))

    print("Extracted line parameters from geometry:")
    print(f"  Zl line: w={w_L*1e6:.2f} um, s={s_L*1e6:.2f} um  ->  "
          f"Zc_L = {Zc_L:.2f} ohm,  eps_eff_L = {eps_eff_L:.3f},  "
          f"quarter-wave L_L = {L_L*1e6:.2f} um @ {f0/1e9:.3f} GHz")
    print(f"  Zh line: w={w_H*1e6:.2f} um, s={s_H*1e6:.2f} um  ->  "
          f"Zc_H = {Zc_H:.2f} ohm,  eps_eff_H = {eps_eff_H:.3f},  "
          f"quarter-wave L_H = {L_H*1e6:.2f} um @ {f0/1e9:.3f} GHz")

    # ---- build the alternating stack --------------------------------------
    stack = build_stack(N_PERIODS, symmetric=SYMMETRIC)
    print(f"\nStack ({len(stack)} sections): {'-'.join(stack)}")

    section_params = {
        'L': (Zc_L, L_L, eps_eff_L),
        'H': (Zc_H, L_H, eps_eff_H),
    }

    # ---- frequency sweep -----------------------------------------------
    # matched to the Ansys sweep (1-11 GHz) for a direct comparison
    freq = np.linspace(2.0e9, 11.0e9, 18001)

    # ---- build ABCD of each segment, then cascade in stack order ---------
    ABCD_list = []
    for label in stack:
        Zc, L, eps_eff = section_params[label]
        ABCD_list.append(line_ABCD(freq, Zc, L, eps_eff, alpha))
    ABCD_total = cascade(ABCD_list)

    # ---- S-parameters and impedance ---------------------------------------
    S11, S12, S21, S22 = ABCD_to_S(ABCD_total, Z0=Z0)
    Zin = input_impedance(ABCD_total, Z0=Z0)
    S11_dB = 20 * np.log10(np.abs(S11))
    S21_dB = 20 * np.log10(np.abs(S21))

    # ---- locate simulated stopband centre (max reflection) ---------------
    idx_stop = np.argmax(S11_dB)
    print(f"\nSimulated stopband centre (max |S11|): {freq[idx_stop]/1e9:.3f} GHz")
    print(f"  |S11| = {np.abs(S11[idx_stop]):.4f} ({S11_dB[idx_stop]:.2f} dB)")
    print(f"  |S21| = {np.abs(S21[idx_stop]):.4f} ({S21_dB[idx_stop]:.2f} dB)")
    print(f"  Zin   = {Zin[idx_stop]:.2f} ohm")

    # ---- -3 dB stopband edges around the peak (bandwidth estimate) -------
    below = np.where(S21_dB <= -3.0)[0]
    if below.size:
        # widest contiguous block containing idx_stop
        left = below[below <= idx_stop]
        right = below[below >= idx_stop]
        f_lo = freq[left[-1]] if left.size else freq[0]
        f_hi = freq[right[0]] if right.size else freq[-1]
        print(f"  Approx. -3 dB stopband: {f_lo/1e9:.3f} - {f_hi/1e9:.3f} GHz "
              f"(BW = {(f_hi - f_lo)/1e9:.3f} GHz)")

    # ---- optional overlay: Ansys (or other EM-sim) S-parameter CSV -------
    # Set ANSYS_CSV_PATH to a CSV exported from Ansys HFSS with frequency
    # + S11/S21 (dB) columns to compare against this ABCD model. Leave as
    # None to skip the overlay.
    ANSYS_CSV_PATH = "D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-comparison-data\Bragg_Resonator_7_Quarters_S_Plots_6GHz.csv"
    ansys_freq_GHz = ansys_S11_dB = ansys_S21_dB = None

    if ANSYS_CSV_PATH is not None:
        import pandas as pd
        df = pd.read_csv(ANSYS_CSV_PATH)
        ansys_freq_GHz = df['Freq [GHz]'].to_numpy()
        ansys_S11_dB = df['dB(S(1,1)) []'].to_numpy()
        ansys_S21_dB = df['dB(S(2,1)) []'].to_numpy()

    # ---- plot ----------------------------------------------------------
    # Two stacked subplots (S11 on top, S21 on bottom) so the curves
    # don't overlap. ABCD model = solid line; Ansys sim = dashed line
    # with markers, in a clearly different color family, so the two
    # sources are easy to tell apart at a glance.
    MODEL_COLOR_S11 = 'tab:orange'
    MODEL_COLOR_S21 = 'tab:blue'
    SIM_COLOR_S11 = 'tab:red'
    SIM_COLOR_S21 = 'tab:green'

    fig, (ax_s11, ax_s21) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # --- S11 subplot ---
    ax_s11.plot(freq / 1e9, S11_dB, color=MODEL_COLOR_S11, lw=1.8,
                label='|S11| (ABCD model)')
    if ansys_freq_GHz is not None:
        ax_s11.plot(ansys_freq_GHz, ansys_S11_dB, color=SIM_COLOR_S11,
                    ls='--', marker='o', ms=3, mfc='none', lw=1.2,
                    alpha=0.85, label='|S11| (Ansys sim)')
    ax_s11.axvline(f0 / 1e9, color='k', ls=':', lw=1, label='design f0')
    ax_s11.set_ylabel('|S11| (dB)')
    ax_s11.set_title(f'{N_PERIODS}-period alternating Bragg resonator '
                      f'({"-".join(stack)}) - Reflection (S11)')
    ax_s11.legend(fontsize=8)
    ax_s11.grid(True, alpha=0.3)

    # --- S21 subplot ---
    ax_s21.plot(freq / 1e9, S21_dB, color=MODEL_COLOR_S21, lw=1.8,
                label='|S21| (ABCD model)')
    if ansys_freq_GHz is not None:
        ax_s21.plot(ansys_freq_GHz, ansys_S21_dB, color=SIM_COLOR_S21,
                    ls='--', marker='s', ms=3, mfc='none', lw=1.2,
                    alpha=0.85, label='|S21| (Ansys sim)')
    ax_s21.axvline(f0 / 1e9, color='k', ls=':', lw=1, label='design f0')
    ax_s21.set_xlabel('Frequency (GHz)')
    ax_s21.set_ylabel('|S21| (dB)')
    ax_s21.set_title('Transmission (S21)')
    ax_s21.legend(fontsize=8)
    ax_s21.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r'D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-outputs\bragg_resonator_7_results_with_sim_data.png', dpi=150)
    print("\nSaved plot")

    # ---- quantitative agreement vs Ansys (RMS error over the overlap) ----
    if ansys_freq_GHz is not None:
        model_S11_on_ansys = np.interp(ansys_freq_GHz, freq / 1e9, S11_dB)
        model_S21_on_ansys = np.interp(ansys_freq_GHz, freq / 1e9, S21_dB)
        rms_S11 = np.sqrt(np.mean((model_S11_on_ansys - ansys_S11_dB) ** 2))
        rms_S21 = np.sqrt(np.mean((model_S21_on_ansys - ansys_S21_dB) ** 2))
        print(f"\nModel vs Ansys RMS error: S11 = {rms_S11:.2f} dB, "
              f"S21 = {rms_S21:.2f} dB (over {ansys_freq_GHz[0]:.2f}-"
              f"{ansys_freq_GHz[-1]:.2f} GHz)")