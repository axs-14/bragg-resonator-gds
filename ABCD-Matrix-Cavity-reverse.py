"""
Inverse fit of a Bragg cavity resonator from S-parameter data.

    [feedline] - [mirror] - cavity (lambda/2) - [mirror] - [feedline]

Mirrors are alternating quarter-wave H/L sections. Any parameter in
PARAM_SPEC can be fixed or fitted; by default only Zc_H is fitted.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

c0 = 2.99792458e8
DB_FLOOR = -80.0


# ---------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------
def line_ABCD(freq, Zc, length, eps_eff, alpha=0.0):
    freq = np.asarray(freq, dtype=float)
    k = 2 * np.pi * freq * np.sqrt(eps_eff) / c0
    GL = (alpha + 1j * k) * length
    ABCD = np.empty((freq.size, 2, 2), dtype=complex)
    ABCD[:, 0, 0] = np.cosh(GL)
    ABCD[:, 0, 1] = Zc * np.sinh(GL)
    ABCD[:, 1, 0] = np.sinh(GL) / Zc
    ABCD[:, 1, 1] = np.cosh(GL)
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


def build_cavity_stack(n_left, n_right, neighbor='H', feedlines=True):
    """Section order, e.g. neighbor='H', 3/3: F-H-L-H-C-H-L-H-F."""
    if neighbor not in ('H', 'L'):
        raise ValueError("neighbor must be 'H' or 'L'")

    def half(n):  # ordered from the cavity outward, alternating H/L
        seq, cur = [], neighbor
        for _ in range(n):
            seq.append(cur)
            cur = 'L' if cur == 'H' else 'H'
        return seq

    core = half(n_left)[::-1] + ['C'] + half(n_right)
    return (['F'] + core + ['F']) if feedlines else core


def model_S(p, freq, stack, f0, eps, feed, Z0=50.0, alpha=0.0):
    """p: Zc_L, Zc_H, Zc_C, s_C.  eps: eps_eff per line.  feed: Zc_F, L_F."""
    L_L = c0 / (4 * f0 * np.sqrt(eps['L']))
    L_H = c0 / (4 * f0 * np.sqrt(eps['H']))
    L_C = p['s_C'] * c0 / (2 * f0 * np.sqrt(eps['C']))
    sections = {
        'L': (p['Zc_L'], L_L, eps['L']),
        'H': (p['Zc_H'], L_H, eps['H']),
        'C': (p['Zc_C'], L_C, eps['C']),
        'F': (feed['Zc_F'], feed['L_F'], eps['F']),
    }
    ABCD_total = cascade([line_ABCD(freq, *sections[k], alpha) for k in stack])
    return ABCD_to_S(ABCD_total, Z0=Z0)


def _to_dB(x):
    return np.maximum(20 * np.log10(np.clip(np.abs(x), 1e-12, None)), DB_FLOOR)


# ---------------------------------------------------------------------
# Fit and uncertainty
# ---------------------------------------------------------------------
def fit(freq, s11_t, s21_t, stack, f0, spec, eps, feed,
        Z0=50.0, alpha=0.0, w_s21=1.5, n_starts=10, seed=0):
    """Multi-start least squares on the S11/S21 dB residuals."""
    s11_t = np.maximum(s11_t, DB_FLOOR)
    s21_t = np.maximum(s21_t, DB_FLOOR)

    names = [n for n, d in spec.items() if d['free']]
    if not names:
        raise ValueError("No free parameters.")
    lo = np.array([spec[n]['bounds'][0] for n in names], float)
    hi = np.array([spec[n]['bounds'][1] for n in names], float)

    def full(x):
        d = {n: spec[n]['value'] for n in spec}
        d.update(zip(names, x))
        return d

    def resid(x):
        S11, S21 = model_S(full(x), freq, stack, f0, eps, feed, Z0, alpha)
        return np.concatenate([_to_dB(S11) - s11_t,
                               w_s21 * (_to_dB(S21) - s21_t)])

    rng = np.random.default_rng(seed)
    starts = [np.clip([spec[n]['value'] for n in names], lo, hi)]
    for _ in range(n_starts - 1):
        starts.append(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    results = []
    for x0 in starts:
        try:
            results.append(least_squares(resid, x0, bounds=(lo, hi), xtol=1e-10,
                                         ftol=1e-10, gtol=1e-10, max_nfev=400))
        except Exception as e:
            print(f"  start {x0} failed: {e}")
    best = min(results, key=lambda r: r.cost)
    return best, names, full, (lo, hi)


def uncertainty(res, names, bounds):
    """
    Covariance = s^2 (J^T J)^-1 from the Jacobian at the solution, with s^2
    taken from the residual scatter. Neighbouring points are correlated, so
    treat the result as a lower bound on the true uncertainty.
    """
    J = res.jac
    m, n = J.shape
    s2 = 2.0 * res.cost / max(m - n, 1)
    cov = s2 * np.linalg.pinv(J.T @ J)
    sig = np.sqrt(np.diag(cov))
    at_bound = [(abs(res.x[i] - bounds[0][i]) < 1e-6 * max(1, bounds[0][i]) or
                 abs(res.x[i] - bounds[1][i]) < 1e-6 * max(1, bounds[1][i]))
                for i in range(n)]
    rms = np.sqrt(2.0 * res.cost / m)
    return sig, cov, rms, at_bound


# =======================================================================
if __name__ == "__main__":

    # ==== EDIT THESE ========================================================
    ANSYS_CSV_PATH = r"D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-comparison-data\AS_Bragg_Resonator_2um_Cavity_8_Quarters_S_Plots_7GHz.csv"
    FREQ_COL, S11_COL, S21_COL = "Freq [GHz]", "dB(S(1,1)) []", "dB(S(2,1)) []"

    f0 = 7e9
    N_LEFT, N_RIGHT = 4,4
    CAVITY_NEIGHBOR = 'L'          # impedance that touches the cavity

    eps = {'L': 6.45, 'H': 6.45, 'C': 6.45, 'F': 6.45}   # eps_eff per line

    # 50 ohm feedlines only add phase when Zc_F equals Z0
    INCLUDE_FEEDLINES = True
    feed = {'Zc_F': 50.0, 'L_F': 1.0e-3}

    # value: fixed value or starting guess;  free: fit this parameter
    PARAM_SPEC = {
        'Zc_L': dict(value=30, free=True, bounds=(25.0, 35.0)),
        'Zc_H': dict(value=90.0,  free=True,  bounds=(85.0, 95.0)),
        'Zc_C': dict(value=108.0,  free=False, bounds=(100.0, 115.0)),
        's_C':  dict(value=1.0,   free=False, bounds=(0.8, 1.2)),  # cavity length scale
    }

    alpha = 0.0                    # Np/m; raise if the data peak is below 0 dB
    Z0 = 50.0
    SAVE_PATH = r'D:\Aryan\Resonator Designs\Bragg Resonator Tests\ABCD-outputs\AS_bragg_resonator_2um_cavity_8_inverse_fit_ratio_result.png'
    # =======================================================================

    stack = build_cavity_stack(N_LEFT, N_RIGHT, CAVITY_NEIGHBOR, INCLUDE_FEEDLINES)
    print(f"Stack ({len(stack)} sections): {'-'.join(stack)}")

    df = pd.read_csv(ANSYS_CSV_PATH)
    freq = df[FREQ_COL].to_numpy() * 1e9
    s11_dB = df[S11_COL].to_numpy()
    s21_dB = df[S21_COL].to_numpy()

    best, names, full, bnds = fit(freq, s11_dB, s21_dB, stack, f0,
                                  PARAM_SPEC, eps, feed, Z0, alpha)
    sig, cov, rms, at_bound = uncertainty(best, names, bnds)
    P = full(best.x)

    # ---- results ------------------------------------------------------------
    iH = names.index('Zc_H')
    ZH, sH = best.x[iH], sig[iH]
    ZL = P['Zc_L']
    ratio = ZH / ZL
    if 'Zc_L' in names:  # include the Zc_H / Zc_L correlation
        iL = names.index('Zc_L')
        s_ratio = ratio * np.sqrt((sH / ZH) ** 2 + (sig[iL] / ZL) ** 2
                                  - 2 * cov[iH, iL] / (ZH * ZL))
    else:
        s_ratio = ratio * sH / ZH
        
    # Extract Zc_C (fitted or fixed)
    if "Zc_C" in names:
      iC = names.index("Zc_C")
      ZC, sC = best.x[iC], sig[iC]
      zc_str = f"{ZC:.2f} +/- {sC:.2f} ohm"
    else:
      ZC = P["Zc_C"]
      zc_str = f"{ZC:.2f} ohm (fixed)"

    print("\nFit result:")
    print(f"  Zc_H       = {ZH:.2f} +/- {sH:.2f} ohm   (Zc_L fixed at {ZL:.2f} ohm)")
    print(f"  Zh / Zl    = {ratio:.3f} +/- {s_ratio:.3f}")
    print(f"  Zc_C         = {zc_str}")
    print(f"  RMS residual = {rms:.2f} dB")
    if any(at_bound):
        print("  WARNING: a fitted parameter hit its bound; result not trustworthy.")

    # ---- plots --------------------------------------------------------------
    freq_fine = np.linspace(freq.min(), freq.max(), 8001)
    S11_f, S21_f = model_S(P, freq_fine, stack, f0, eps, feed, Z0, alpha)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(freq / 1e9, s11_dB, 'o', ms=3, mfc='none', color='tab:red', label='Ansys S11')
    ax1.plot(freq_fine / 1e9, 20 * np.log10(np.abs(S11_f) + 1e-15),
             color='tab:orange', label='Fitted model S11')
    ax1.set_ylabel('|S11| (dB)'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title(f'Bragg cavity inverse fit ({"-".join(stack)})')

    ax2.plot(freq / 1e9, s21_dB, 's', ms=3, mfc='none', color='tab:green', label='Ansys S21')
    ax2.plot(freq_fine / 1e9, 20 * np.log10(np.abs(S21_f) + 1e-15),
             color='tab:blue', label='Fitted model S21')
    ax2.set_xlabel('Frequency (GHz)'); ax2.set_ylabel('|S21| (dB)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150)
    print(f"\nSaved plot to {SAVE_PATH}")