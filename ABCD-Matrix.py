"""
ABCD Matrix simulation of a PBG (photonic bandgap) resonator built from
cascaded coplanar transmission-line segments T_A, T_B, T_C.

Physics (from Pozar, "Microwave Engineering"):
    - Each line segment (length L, characteristic impedance Zc, complex
      propagation constant Gamma = alpha + i*k) is represented by a
      2x2 ABCD matrix:

            [A B]   [ cosh(GL)      Zc*sinh(GL) ]
            [C D] = [ sinh(GL)/Zc   cosh(GL)     ]

    - The whole device's ABCD matrix is the ordered product of the
      individual segment matrices.

    - Device layout used in this report:
            (T_A - T_B) x 5  -  T_C  -  (T_B - T_A) x 5

    - Once the total ABCD matrix is known it is converted to the
      S-parameter matrix (referenced to Z0, usually 50 ohm):

        S11 = (A + B/Zc - C*Z0 - D) / (A + B/Zc + C*Z0 + D)
        S21 = 2 / (A + B/Zc + C*Z0 + D)
        S12 = 2*(AD - BC) / (A + B/Zc + C*Z0 + D)
        S22 = (-A + B/Zc - C*Z0 + D) / (A + B/Zc + C*Z0 + D)

    - The input impedance looking into the network when terminated in
      Z0 is:
            Zin = (A*Z0 + B) / (C*Z0 + D)

Notes on parameters you will likely want to tune:
    - eps_eff : effective permittivity of the coplanar waveguide. This
      sets the propagation constant k = 2*pi*f*sqrt(eps_eff)/c0. The
      earlier IDC section of the report used eps_eff = 6.45 for a
      silicon substrate CPW, so that is used as the default here --
      replace it with your own extracted/simulated value if you have one.
    - alpha : attenuation constant (Np/m). Zero by default (lossless
      line); set a small positive number if you want to include
      conductor/dielectric loss.
    - Zc_A, Zc_B, Zc_C, L_A, L_B, L_C : geometry/impedance of the three
      line types, taken directly from the figure (Z_TA = 92 ohm,
      L_TA = 4.2 mm; Z_TB = 34 ohm, L_TB = 4.2 mm; Z_TC = 50 ohm,
      L_TC = 6 mm).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

c0 = 2.99792458e8  # speed of light in vacuum, m/s


# ---------------------------------------------------------------------
# 1. Single transmission-line segment -> 2x2 ABCD matrix (vectorised
#    over an array of frequencies)
# ---------------------------------------------------------------------
def line_ABCD(freq, Zc, length, eps_eff, alpha=0.0):
    """
    Returns the ABCD matrix of a transmission line for each frequency
    in `freq`.

    Parameters
    ----------
    freq     : array_like, frequencies in Hz
    Zc       : characteristic impedance of the line, ohm
    length   : physical length of the line, m
    eps_eff  : effective relative permittivity (sets phase velocity)
    alpha    : attenuation constant, Np/m (default 0 -> lossless)

    Returns
    -------
    ABCD : ndarray of shape (len(freq), 2, 2), complex
    """
    freq = np.asarray(freq, dtype=float)
    k = 2 * np.pi * freq * np.sqrt(eps_eff) / c0        # propagation constant
    Gamma = alpha + 1j * k                                # Gamma = alpha + i*k
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
# 2. Cascade a list of ABCD matrices (matrix multiplication, in order,
#    at every frequency point)
# ---------------------------------------------------------------------
def cascade(ABCD_list):
    """Multiply a sequence of (Nfreq, 2, 2) ABCD arrays, in order."""
    total = ABCD_list[0].copy()
    for M in ABCD_list[1:]:
        total = np.einsum('nij,njk->nik', total, M)
    return total


# ---------------------------------------------------------------------
# 3. ABCD -> S-parameters
# ---------------------------------------------------------------------
def ABCD_to_S(ABCD, Z0=50.0):
    A = ABCD[:, 0, 0]
    B = ABCD[:, 0, 1]
    C = ABCD[:, 1, 0]
    D = ABCD[:, 1, 1]

    denom = A + B / Z0 + C * Z0 + D

    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S21 = 2.0 / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S22 = (-A + B / Z0 - C * Z0 + D) / denom

    return S11, S12, S21, S22


# ---------------------------------------------------------------------
# 4. Input impedance of the network when terminated in Z0
# ---------------------------------------------------------------------
def input_impedance(ABCD, Z0=50.0):
    A = ABCD[:, 0, 0]
    B = ABCD[:, 0, 1]
    C = ABCD[:, 1, 0]
    D = ABCD[:, 1, 1]
    return (A * Z0 + B) / (C * Z0 + D)


# =======================================================================
# Build the PBG device:  (T_A - T_B) x 5  -  T_C  -  (T_B - T_A) x 5
# =======================================================================
if __name__ == "__main__":

    # ---- design parameters (from the figure) --------------------------
    Z_TA, L_TA = 92.0, 4.2e-3   # ohm, m  (Bragg mirror line A)
    Z_TB, L_TB = 34.0, 4.2e-3   # ohm, m  (Bragg mirror line B)
    Z_TC, L_TC = 50.0, 6.0e-3   # ohm, m  (defect line)

    eps_eff = 6.45              # effective permittivity (edit to match your CPW)
    alpha = 0.0                 # Np/m, set >0 to include loss
    Z0 = 50.0                   # port/reference impedance, ohm

    # ---- frequency sweep -----------------------------------------------
    freq = np.linspace(1e9, 20e9, 4001)   # 1 - 20 GHz

    # ---- ABCD matrices of each segment, per frequency -------------------
    M_TA = line_ABCD(freq, Z_TA, L_TA, eps_eff, alpha)
    M_TB = line_ABCD(freq, Z_TB, L_TB, eps_eff, alpha)
    M_TC = line_ABCD(freq, Z_TC, L_TC, eps_eff, alpha)

    # ---- assemble the cascade: (TA-TB)x5 - TC - (TB-TA)x5 ---------------
    mirror_left  = [M_TA, M_TB] * 5          # TA,TB,TA,TB,...  (5 pairs)
    mirror_right = [M_TB, M_TA] * 5          # TB,TA,TB,TA,...  (5 pairs)
    full_stack = mirror_left + [M_TC] + mirror_right

    ABCD_total = cascade(full_stack)

    # ---- convert to S-parameters and impedance ---------------------------
    S11, S12, S21, S22 = ABCD_to_S(ABCD_total, Z0=Z0)
    Zin = input_impedance(ABCD_total, Z0=Z0)

    S11_dB = 20 * np.log10(np.abs(S11))
    S21_dB = 20 * np.log10(np.abs(S21))

    # ---- find the stopband(s) and any narrow resonance inside them ------
    # The Bragg mirrors (T_A/T_B pairs) create one or more broad stopbands
    # (|S11| -> 1, |S21| -> 0). If the T_C defect places a resonance inside
    # one of those gaps it shows up as a single sharp, narrow S21 peak
    # sitting on top of the otherwise deep rejection -- much narrower than
    # the ordinary Fabry-Perot ripple that exists everywhere else in the
    # passbands. We isolate candidate defect peaks by requiring both a
    # narrow linewidth (high Q) AND a locally deep surrounding rejection.
    stopband_mask = S11_dB > -3          # deep-reflection region
    peaks, _ = find_peaks(S21_dB, height=-10)
    widths = peak_widths(S21_dB, peaks, rel_height=0.5)[0]

    # keep only peaks whose immediate neighbourhood (+/- ~10 points) is
    # mostly a stopband, i.e. genuinely embedded resonances, not passband
    # ripple
    def in_stopband_neighbourhood(idx, span=15):
        lo, hi = max(0, idx - span), min(len(freq), idx + span)
        return stopband_mask[lo:idx].mean() > 0.4 or stopband_mask[idx:hi].mean() > 0.4

    candidates = [i for i in range(len(peaks)) if in_stopband_neighbourhood(peaks[i])]

    print(f"Ports referenced to Z0 = {Z0:.0f} ohm, "
          f"eps_eff = {eps_eff}, sweep {freq[0]/1e9:.1f}-{freq[-1]/1e9:.1f} GHz\n")

    f_defect_idx = None
    if candidates:
        best = candidates[int(np.argmin(widths[candidates]))]
        f_defect_idx = peaks[best]
        print(f"Candidate defect (cavity) resonance found at "
              f"{freq[f_defect_idx]/1e9:.4f} GHz")
        print(f"  |S21| = {np.abs(S21[f_defect_idx]):.4f} ({S21_dB[f_defect_idx]:.2f} dB)")
        print(f"  |S11| = {np.abs(S11[f_defect_idx]):.4f} ({S11_dB[f_defect_idx]:.2f} dB)")
        print(f"  Zin   = {Zin[f_defect_idx]:.2f} ohm")
    else:
        print("No narrow resonance embedded inside a stopband was found "
              "in this frequency sweep -- inspect the plot and/or widen "
              "the sweep range, or adjust eps_eff/geometry.")

    # ---- plot S-parameters --------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(freq / 1e9, S21_dB, label='|S21| (transmission)')
    axes[0].plot(freq / 1e9, S11_dB, label='|S11| (reflection)')
    if f_defect_idx is not None:
        axes[0].axvline(freq[f_defect_idx] / 1e9, color='k', ls='--', lw=0.8,
                         label='defect mode')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('PBG resonator S-parameters (ABCD matrix method)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freq / 1e9, np.real(Zin), label='Re(Zin)')
    axes[1].plot(freq / 1e9, np.imag(Zin), label='Im(Zin)')
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('Impedance (ohm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pbg_abcd_results.png', dpi=150)
    print("\nSaved plot to pbg_abcd_results.png")