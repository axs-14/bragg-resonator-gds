"""
Inverse-fit the Bragg stack's low characteristic impedance (Zc_L) 
and impedance ratio (r = Zc_H / Zc_L) from measured/simulated S-parameter data.

Topology:
  1 period  = H - L - H
  2 periods = H - L - H - L - H
  N periods = H + N * [L, H]
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

c0 = 2.99792458e8


# ---------------------------------------------------------------------
# Forward-model building blocks
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
    ABCD[:, 0, 0], ABCD[:, 0, 1] = A, B
    ABCD[:, 1, 0], ABCD[:, 1, 1] = C, D
    return ABCD


def cascade(ABCD_list):
    total = ABCD_list[0].copy()
    for M in ABCD_list[1:]:
        total = np.einsum('nij,njk->nik', total, M)
    return total


def ABCD_to_S(ABCD, Z0=50.0):
    A, B, C, D = (ABCD[:, 0, 0], ABCD[:, 0, 1], ABCD[:, 1, 0], ABCD[:, 1, 1])
    denom = A + B / Z0 + C * Z0 + D
    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S21 = 2.0 / denom
    return S11, S21


def build_stack(N_periods):
    """
    Builds symmetric Bragg stack topology:
    N=1 -> ['H', 'L', 'H']
    N=2 -> ['H', 'L', 'H', 'L', 'H']
    """
    stack = ['H']
    for _ in range(N_periods):
        stack += ['L', 'H']
    return stack


def model_S_ratio(params, freq, stack, f0, Z0=50.0, alpha=0.0):
    """
    Parametrized forward model using (Zc_L, r, eps_eff_L, eps_eff_H)
    where r = Zc_H / Zc_L.
    """
    Zc_L, r, eps_eff_L, eps_eff_H = params
    Zc_H = Zc_L * r

    L_L = c0 / (4 * f0 * np.sqrt(eps_eff_L))
    L_H = c0 / (4 * f0 * np.sqrt(eps_eff_H))
    section_params = {'L': (Zc_L, L_L, eps_eff_L), 'H': (Zc_H, L_H, eps_eff_H)}

    ABCD_list = [line_ABCD(freq, *section_params[label], alpha) for label in stack]
    ABCD_total = cascade(ABCD_list)
    return ABCD_to_S(ABCD_total, Z0=Z0)


# ---------------------------------------------------------------------
# Direct Ratio Fitting Function
# ---------------------------------------------------------------------
def fit_ratio_and_Zl(freq_Hz, s11_target_dB, s21_target_dB, stack, f0,
                       eps_eff_L, eps_eff_H, x0=(30.0, 3.0),
                       bounds=None, Z0=50.0):
    if bounds is None:
        bounds = ([5.0, 1.05], [200.0, 20.0])

    def resid(p):
        Zc_L, r = p[0], p[1]
        S11, S21 = model_S_ratio([Zc_L, r, eps_eff_L, eps_eff_H], freq_Hz, stack, f0, Z0=Z0)
        s11_dB = 20 * np.log10(np.clip(np.abs(S11), 1e-12, None))
        s21_dB = 20 * np.log10(np.clip(np.abs(S21), 1e-12, None))
        return np.concatenate([(s11_dB - s11_target_dB), 1.5 * (s21_dB - s21_target_dB)])

    result = least_squares(
        resid, x0, bounds=bounds,
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
        max_nfev=500
    )
    return result


# =======================================================================
if __name__ == "__main__":
    ANSYS_CSV_PATH = r"D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-comparison-data\Bragg_Resonator_7_Quarters_S_Plots_6GHz.csv"
    FREQ_COL, S11_COL, S21_COL = "Freq [GHz]", "dB(S(1,1)) []", "dB(S(2,1)) []"

    f0 = 6e9           # Design quarter-wave frequency
    N_PERIODS = 3      # 1 period -> ['H', 'L', 'H']

    eps_eff_L_fixed = 6.45
    eps_eff_H_fixed = 6.45
    Z0 = 50.0

    stack = build_stack(N_PERIODS)
    print(f"Fitting stack: {'-'.join(stack)}")  # Will print: H-L-H for N=1

    df = pd.read_csv(ANSYS_CSV_PATH)
    freq = df[FREQ_COL].to_numpy() * 1e9
    s11_dB = df[S11_COL].to_numpy()
    s21_dB = df[S21_COL].to_numpy()

    # Optimization pass starting from nominal guess (Zc_L = 30 ohms, ratio r = 3.0)
    fit_res = fit_ratio_and_Zl(
        freq, s11_dB, s21_dB, stack, f0,
        eps_eff_L_fixed, eps_eff_H_fixed, Z0=Z0
    )

    Zc_L_fit, r_fit = fit_res.x
    Zc_H_fit = Zc_L_fit * r_fit

    print("\nFit Results:")
    print(f"  Low Impedance  (Zc_L)  = {Zc_L_fit:.2f} ohm")
    print(f"  High Impedance (Zc_H)  = {Zc_H_fit:.2f} ohm")
    print(f"  Impedance Ratio (r)    = {r_fit:.3f}")
    print(f"  Optimizer Residual Cost = {fit_res.cost:.4f}")

    # Plot fit comparison
    freq_fine = np.linspace(freq.min(), freq.max(), 4001)
    S11_fit, S21_fit = model_S_ratio([Zc_L_fit, r_fit, eps_eff_L_fixed, eps_eff_H_fixed], freq_fine, stack, f0, Z0=Z0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(freq / 1e9, s11_dB, 'o', ms=3, mfc='none', color='tab:red', label='Ansys S11')
    ax1.plot(freq_fine / 1e9, 20 * np.log10(np.abs(S11_fit)), color='tab:orange', label='Fitted model S11')
    ax1.set_ylabel('|S11| (dB)'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(freq / 1e9, s21_dB, 's', ms=3, mfc='none', color='tab:green', label='Ansys S21')
    ax2.plot(freq_fine / 1e9, 20 * np.log10(np.abs(S21_fit)), color='tab:blue', label='Fitted model S21')
    ax2.set_xlabel('Frequency (GHz)'); ax2.set_ylabel('|S21| (dB)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-outputs\bragg_resonator_7_inverse_fit_ratio_result_.png", dpi=150)
    print("\nSaved plot to bragg_inverse_fit_ratio_result.png")