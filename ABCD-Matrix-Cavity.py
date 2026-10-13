"""
ABCD-matrix simulation of a Bragg CAVITY resonator:

    [feedline] - [left quarter-wave mirror] - CAVITY (lambda/2) - [right quarter-wave mirror] - [feedline]

Mirrors are alternating high/low impedance quarter-wave sections (Zh, Zl).
The cavity is a half-wave section at f0 with its own free dimensions
(w_C, s_C). Either H or L can be the impedance that touches the cavity.
A 50 ohm launcher/feedline section is added at each port.
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

    hi = k >= 0.707
    lo = ~hi

    ratio[hi] = (1 / np.pi) * np.log(2 * (1 + np.sqrt(k[hi])) / (1 - np.sqrt(k[hi])))
    ratio[lo] = np.pi / np.log(2 * (1 + np.sqrt(kp[lo])) / (1 - np.sqrt(kp[lo])))
    return ratio


# ---------------------------------------------------------------------
# 1a. Coplanar waveguide (CPW)
# ---------------------------------------------------------------------
def cpw_zc_epseff(w, s, eps_r, h_sub=None):
    w = np.atleast_1d(np.asarray(w, dtype=float))
    s = np.atleast_1d(np.asarray(s, dtype=float))

    k = w / (w + 2 * s)
    kke = kk_ratio(np.array([k[0]]))[0] if w.size == 1 else kk_ratio(k)

    if h_sub is None:
        eps_eff = np.full_like(k, (eps_r + 1) / 2.0)
    else:
        k1 = np.sinh(np.pi * w / (4 * h_sub)) / np.sinh(np.pi * (w + 2 * s) / (4 * h_sub))
        kk1 = kk_ratio(np.array([k1[0]]))[0] if w.size == 1 else kk_ratio(k1)
        eps_eff = 1 + (eps_r - 1) / 2.0 * (kk1 / kke)

    Zc = 30 * np.pi / np.sqrt(eps_eff) / kke
    return (Zc.item(), eps_eff.item()) if Zc.size == 1 else (Zc, eps_eff)


# ---------------------------------------------------------------------
# 1b. Microstrip (Hammerstad-Jensen)
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
# 1c. Geometry -> (Zc, eps_eff)
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
# 2. Transmission-line segment -> ABCD vs frequency
# ---------------------------------------------------------------------
def line_ABCD(freq, Zc, length, eps_eff, alpha=0.0):
    freq = np.asarray(freq, dtype=float)
    k = 2 * np.pi * freq * np.sqrt(eps_eff) / c0
    Gamma = alpha + 1j * k
    GL = Gamma * length

    ABCD = np.empty((freq.size, 2, 2), dtype=complex)
    ABCD[:, 0, 0] = np.cosh(GL)
    ABCD[:, 0, 1] = Zc * np.sinh(GL)
    ABCD[:, 1, 0] = np.sinh(GL) / Zc
    ABCD[:, 1, 1] = np.cosh(GL)
    return ABCD


# ---------------------------------------------------------------------
# 3. Cascade
# ---------------------------------------------------------------------
def cascade(ABCD_list):
    total = ABCD_list[0].copy()
    for M in ABCD_list[1:]:
        total = np.einsum('nij,njk->nik', total, M)
    return total


# ---------------------------------------------------------------------
# 4. ABCD -> S-parameters, input impedance
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
# 5. Build:  [left mirror] - CAVITY - [right mirror]
# ---------------------------------------------------------------------
def build_cavity_stack(n_left, n_right, neighbor='H'):
    """
    n_left, n_right : number of quarter-wave mirror sections on each side
    neighbor        : 'H' or 'L' -> impedance that touches the cavity.
                      Sections alternate moving outward from the cavity.

    neighbor='H', n_left=n_right=3:  L-H... reads  H-L-H - C - H-L-H
    neighbor='L', n_left=n_right=3:  L-H-L - C - L-H-L
    """
    if neighbor not in ('H', 'L'):
        raise ValueError("neighbor must be 'H' or 'L'")

    def half(n):
        seq, cur = [], neighbor
        for _ in range(n):
            seq.append(cur)
            cur = 'L' if cur == 'H' else 'H'
        return seq                       # ordered cavity -> outward

    return half(n_left)[::-1] + ['C'] + half(n_right)


# =======================================================================
if __name__ == "__main__":

    # ==== EDIT THESE ========================================================
    LINE_KIND = "cpw"
    eps_r = 11.9
    h_sub = 300e-6
    f0 = 7e9                  # mirrors = lambda/4, cavity = lambda/2

    N_LEFT = 6
    N_RIGHT = 6
    CAVITY_NEIGHBOR = 'L'     # 'H' or 'L'

    w_L, s_L = 42e-6, 4e-6    # low-impedance line (Zl)
    w_H, s_H = 4e-6, 23e-6    # high-impedance line (Zh)
    w_C, s_C = 2e-6, 24e-6    # cavity line (free parameters)

    # 50 ohm launcher / feedline at each port
    INCLUDE_FEEDLINES = True
    Zc_F = 50.0
    eps_eff_F = 6.45          # feedline eps_eff (from its geometry)
    L_F = 50.0e-6              # feedline length, m

    alpha = 0.0
    Z0 = 50.0

    ANSYS_CSV_PATH = r"D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-comparison-data\AS_Bragg_Resonator_2um_Cavity_12_Bigger_Vacuum_Quarters_S_Plots_7GHz.csv"
    SAVE_PATH = r'D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-outputs\AS_bragg_resonator_2um_cavity_12_1mm_vacuum_results_with_sim_data.png'
    # =======================================================================

    Zc_L, eps_eff_L = zc_and_epseff(w_L, s_L, eps_r, h_sub, kind=LINE_KIND)
    Zc_H, eps_eff_H = zc_and_epseff(w_H, s_H, eps_r, h_sub, kind=LINE_KIND)
    Zc_C, eps_eff_C = zc_and_epseff(w_C, s_C, eps_r, h_sub, kind=LINE_KIND)

    L_L = c0 / (4 * f0 * np.sqrt(eps_eff_L))
    L_H = c0 / (4 * f0 * np.sqrt(eps_eff_H))
    L_C = c0 / (2 * f0 * np.sqrt(eps_eff_C))

    print("Extracted line parameters:")
    for name, w, s, Zc, ee, Ln, frac in [
            ("Zl  ", w_L, s_L, Zc_L, eps_eff_L, L_L, "lambda/4"),
            ("Zh  ", w_H, s_H, Zc_H, eps_eff_H, L_H, "lambda/4"),
            ("Cav ", w_C, s_C, Zc_C, eps_eff_C, L_C, "lambda/2")]:
        print(f"  {name}: w={w*1e6:.2f} um, s={s*1e6:.2f} um -> Zc = {Zc:.2f} ohm, "
              f"eps_eff = {ee:.3f}, length ({frac}) = {Ln*1e6:.2f} um")
    if INCLUDE_FEEDLINES:
        print(f"  Feed: Zc = {Zc_F:.2f} ohm, eps_eff = {eps_eff_F:.3f}, "
              f"length = {L_F*1e6:.2f} um (each port)")

    stack = build_cavity_stack(N_LEFT, N_RIGHT, CAVITY_NEIGHBOR)
    if INCLUDE_FEEDLINES:
        stack = ['F'] + stack + ['F']
    print(f"\nStack ({len(stack)} sections): {'-'.join(stack)}")

    section_params = {
        'L': (Zc_L, L_L, eps_eff_L),
        'H': (Zc_H, L_H, eps_eff_H),
        'C': (Zc_C, L_C, eps_eff_C),
        'F': (Zc_F, L_F, eps_eff_F),
    }

    freq = np.linspace(2.0e9, 11.0e9, 18001)

    ABCD_list = []
    for label in stack:
        Zc, L, eps_eff = section_params[label]
        ABCD_list.append(line_ABCD(freq, Zc, L, eps_eff, alpha))
    ABCD_total = cascade(ABCD_list)

    S11, S12, S21, S22 = ABCD_to_S(ABCD_total, Z0=Z0)
    Zin = input_impedance(ABCD_total, Z0=Z0)
    S11_dB = 20 * np.log10(np.abs(S11) + 1e-15)
    S21_dB = 20 * np.log10(np.abs(S21) + 1e-15)

    # ---- cavity resonance: transmission peak inside the stopband ----------
    win = (freq > 0.7 * f0) & (freq < 1.3 * f0)
    peaks, _ = find_peaks(S21_dB[win], height=-40)
    if peaks.size:
        pk = np.where(win)[0][peaks[np.argmin(np.abs(freq[win][peaks] - f0))]]
        f_res, peak_dB = freq[pk], S21_dB[pk]
        thr = peak_dB - 3.0
        l = pk
        while l > 0 and S21_dB[l] > thr:
            l -= 1
        r = pk
        while r < freq.size - 1 and S21_dB[r] > thr:
            r += 1
        bw = freq[r] - freq[l]
        print(f"\nCavity resonance: {f_res/1e9:.4f} GHz, peak |S21| = {peak_dB:.2f} dB")
        print(f"  -3 dB linewidth = {bw/1e6:.2f} MHz  ->  loaded Q ~ {f_res/bw:.0f}")
    else:
        print("\nNo transmission peak found near f0 (check mirror strength / geometry).")

    # ---- optional Ansys overlay -------------------------------------------
    ansys_freq_GHz = ansys_S11_dB = ansys_S21_dB = None
    if ANSYS_CSV_PATH is not None:
        import pandas as pd
        df = pd.read_csv(ANSYS_CSV_PATH)
        ansys_freq_GHz = df['Freq [GHz]'].to_numpy()
        ansys_S11_dB = df['dB(S(1,1)) []'].to_numpy()
        ansys_S21_dB = df['dB(S(2,1)) []'].to_numpy()

    # ---- plot ---------------------------------------------------------------
    fig, (ax_s11, ax_s21) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_s11.plot(freq / 1e9, S11_dB, color='tab:orange', lw=1.8, label='|S11| (ABCD)')
    ax_s21.plot(freq / 1e9, S21_dB, color='tab:blue', lw=1.8, label='|S21| (ABCD)')
    if ansys_freq_GHz is not None:
        ax_s11.plot(ansys_freq_GHz, ansys_S11_dB, color='tab:red', ls='--',
                    marker='o', ms=3, mfc='none', lw=1.2, alpha=0.85, label='|S11| (Ansys)')
        ax_s21.plot(ansys_freq_GHz, ansys_S21_dB, color='tab:green', ls='--',
                    marker='s', ms=3, mfc='none', lw=1.2, alpha=0.85, label='|S21| (Ansys)')
    for ax in (ax_s11, ax_s21):
        ax.axvline(f0 / 1e9, color='k', ls=':', lw=1, label='design f0')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax_s11.set_ylabel('|S11| (dB)')
    ax_s11.set_title(f'Bragg cavity resonator ({"-".join(stack)}) - Reflection')
    ax_s21.set_xlabel('Frequency (GHz)')
    ax_s21.set_ylabel('|S21| (dB)')
    ax_s21.set_title('Transmission')

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150)
    print("\nSaved plot")

    # ---- RMS agreement vs Ansys -------------------------------------------
    if ansys_freq_GHz is not None:
        m11 = np.interp(ansys_freq_GHz, freq / 1e9, S11_dB)
        m21 = np.interp(ansys_freq_GHz, freq / 1e9, S21_dB)
        rms11 = np.sqrt(np.mean((m11 - ansys_S11_dB) ** 2))
        rms21 = np.sqrt(np.mean((m21 - ansys_S21_dB) ** 2))
        print(f"\nModel vs Ansys RMS error: S11 = {rms11:.2f} dB, S21 = {rms21:.2f} dB "
              f"(over {ansys_freq_GHz[0]:.2f}-{ansys_freq_GHz[-1]:.2f} GHz)")