"""
Fourier Psychology — Spectral Analysis Engine
==============================================
Applies FFT-based signal analysis to hourly activity time-series
and extracts four spectral metrics that quantify behavioural structure:

* **Spectral Entropy (Hₙ)** — order vs chaos in behaviour (0 = ordered, 1 = chaotic)
* **Spectral Centroid (Cₛ)** — governing timescale of behaviour
* **Harmonic-to-Noise Ratio (HNR)** — proportion of structured pattern vs noise
* **Spectral Slope (β)** — balance between stability and reactivity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_SPACING_HOURS: float = 1.0

# Population baseline — seeded from synthetic data, calibrate with real data.
# Calibrated from 100 synthetic developers (180-day signals, seed=42).
# Re-run `python engine.py` to recalibrate from real data.
POPULATION_BASELINE: Dict[str, Dict[str, float]] = {
    "spectral_entropy": {"mean": 0.4816, "std": 0.0681},
    "spectral_centroid_hz": {"mean": 0.112264, "std": 0.010625},
    "hnr_dB": {"mean": 1.92, "std": 1.15},
    "spectral_slope": {"mean": 0.0927, "std": 0.054},
}


# ---------------------------------------------------------------------------
# Core FFT computation
# ---------------------------------------------------------------------------

def compute_fft(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run real-valued FFT on an hourly time-series.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of hourly event counts (real-valued).

    Returns
    -------
    freqs : np.ndarray
        Frequency bins in cycles per hour.
    coeffs : np.ndarray
        Complex Fourier coefficients from ``rfft``.
    power : np.ndarray
        Power spectrum (|coeffs|²).
    """
    n = len(signal)
    coeffs = rfft(signal)
    freqs = rfftfreq(n, d=SAMPLE_SPACING_HOURS)  # cycles per hour
    power = np.abs(coeffs) ** 2
    return freqs, coeffs, power


# ---------------------------------------------------------------------------
# Metric 1: Spectral Entropy (Hₙ)
# ---------------------------------------------------------------------------

def spectral_entropy(freqs: np.ndarray, power: np.ndarray) -> float:
    """Normalised spectral entropy Hₙ ∈ [0, 1].

    Measures how evenly energy is distributed across frequencies.
    Low entropy (→ 0) means behaviour is dominated by a few strong rhythms.
    High entropy (→ 1) means behaviour has no dominant pattern (white noise).

    Math:
        p(fₖ) = P(fₖ) / Σ P(fₖ)           (normalise to probability)
        H = −Σ p(fₖ) · log₂(p(fₖ))        (Shannon entropy)
        Hₙ = H / log₂(N)                   (normalise to [0, 1])
    """
    ac_mask = freqs > 0
    p = power[ac_mask]
    N = len(p)
    total = p.sum()
    if total == 0 or N <= 1:
        return 0.0
    p_norm = p / total
    nonzero = p_norm > 0
    H = -np.sum(p_norm[nonzero] * np.log2(p_norm[nonzero]))
    return float(H / np.log2(N))


# ---------------------------------------------------------------------------
# Metric 2: Spectral Centroid (Cₛ)
# ---------------------------------------------------------------------------

def spectral_centroid(freqs: np.ndarray, power: np.ndarray) -> float:
    """Spectral centroid in cycles per hour.

    The power-weighted average frequency — the "centre of mass" of the
    spectrum. Indicates the governing frequency of the behaviour.
    Convert to timescale via: governing_period = 1 / centroid (hours).
    """
    ac_mask = freqs > 0
    f = freqs[ac_mask]
    p = power[ac_mask]
    total = p.sum()
    if total == 0:
        return 0.0
    return float(np.sum(f * p) / total)


# ---------------------------------------------------------------------------
# Metric 3: Harmonic-to-Noise Ratio (HNR)
# ---------------------------------------------------------------------------

def harmonic_to_noise_ratio(freqs: np.ndarray, power: np.ndarray) -> float:
    """Harmonic-to-noise ratio in decibels.

    Identifies spectral peaks (power > mean + 2σ) as harmonic (structured)
    energy and everything else as noise. Higher dB means behaviour is
    dominated by repeating patterns rather than random fluctuation.
    """
    ac_mask = freqs > 0
    p = power[ac_mask]
    if len(p) == 0 or p.sum() == 0:
        return 0.0
    mean_p = np.mean(p)
    std_p = np.std(p)
    threshold = mean_p + 2.0 * std_p
    harmonic_mask = p > threshold
    E_harmonic = p[harmonic_mask].sum()
    E_noise = p[~harmonic_mask].sum()
    if E_noise == 0:
        return 40.0  # cap — all energy is harmonic
    if E_harmonic == 0:
        return -10.0  # floor — no detectable peaks
    return float(10.0 * np.log10(E_harmonic / E_noise))


# ---------------------------------------------------------------------------
# Metric 4: Spectral Slope (β)
# ---------------------------------------------------------------------------

def spectral_slope(freqs: np.ndarray, power: np.ndarray) -> float:
    """Spectral slope β from log-log linear regression.

    Fits log₁₀(P) = −β · log₁₀(f) + c across the AC spectrum.
    β ≈ 0 → white noise (chaotically reactive)
    β ≈ 1 → pink / 1/f noise (adaptively flexible)
    β ≈ 2 → brown / 1/f² noise (rigidly stable)
    """
    ac_mask = freqs > 0
    f = freqs[ac_mask]
    p = power[ac_mask]
    valid = p > 0
    f = f[valid]
    p = p[valid]
    if len(f) < 2:
        return 0.0
    log_f = np.log10(f)
    log_p = np.log10(p)
    slope, _ = np.polyfit(log_f, log_p, 1)
    return float(-slope)


# ---------------------------------------------------------------------------
# Peak detection — dominant periods
# ---------------------------------------------------------------------------

def find_dominant_periods(
    freqs: np.ndarray,
    power: np.ndarray,
    n_peaks: int = 5,
) -> List[Dict]:
    """Identify the strongest spectral peaks and convert to periods.

    Returns a list of dicts sorted by power (descending), each with:
        period_hours, label, power.
    """
    ac_mask = freqs > 0
    f = freqs[ac_mask]
    p = power[ac_mask]
    if len(p) == 0:
        return []

    peak_indices, _ = find_peaks(p, prominence=np.max(p) * 0.01)

    if len(peak_indices) == 0:
        # Fallback: top bins by raw power
        peak_indices = np.argsort(p)[-n_peaks:][::-1]

    # Sort by power descending, take top n
    sorted_idx = peak_indices[np.argsort(p[peak_indices])[::-1]]
    top = sorted_idx[:n_peaks]

    results = []
    for idx in top:
        freq = f[idx]
        if freq <= 0:
            continue
        period_hours = 1.0 / freq
        results.append({
            "period_hours": round(float(period_hours), 1),
            "label": _format_period(period_hours),
            "power": float(p[idx]),
        })
    return results


def _format_period(hours: float) -> str:
    """Convert a period in hours to a human-readable label."""
    if hours < 48:
        return f"~{hours:.0f} hours"
    days = hours / 24
    return f"~{days:.1f} days"


# ---------------------------------------------------------------------------
# Divergence from population baseline
# ---------------------------------------------------------------------------

def compute_divergence(
    entropy: float,
    centroid_hz: float,
    hnr_dB: float,
    slope: float,
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """Compute z-scores and composite divergence D.

    D = sqrt(z_entropy² + z_centroid² + z_hnr² + z_slope²),
    expressed in standard deviations from the population mean.
    """
    if baseline is None:
        baseline = POPULATION_BASELINE

    def _z(value: float, key: str) -> float:
        s = baseline[key]["std"]
        return (value - baseline[key]["mean"]) / s if s > 0 else 0.0

    z_e = _z(entropy, "spectral_entropy")
    z_c = _z(centroid_hz, "spectral_centroid_hz")
    z_h = _z(hnr_dB, "hnr_dB")
    z_s = _z(slope, "spectral_slope")
    D = float(np.sqrt(z_e**2 + z_c**2 + z_h**2 + z_s**2))

    return {
        "z_entropy": round(z_e, 3),
        "z_centroid": round(z_c, 3),
        "z_hnr": round(z_h, 3),
        "z_slope": round(z_s, 3),
        "composite_D": round(D, 3),
    }


# ---------------------------------------------------------------------------
# Profile dataclass and main entry point
# ---------------------------------------------------------------------------

@dataclass
class SpectralProfile:
    """Complete spectral analysis result for one user."""
    spectral_entropy: float
    spectral_centroid_hz: float
    governing_timescale_hours: float
    governing_timescale_label: str
    harmonic_to_noise_ratio_dB: float
    spectral_slope: float
    dominant_periods: List[Dict]
    divergence: Dict
    power_spectrum: Dict
    metadata: Dict


def analyse(
    signal: np.ndarray,
    start_date: str = "",
    end_date: str = "",
    baseline: Optional[Dict] = None,
) -> SpectralProfile:
    """Run full spectral analysis on an hourly event-count signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D array where element *i* = number of events in hour *i*.
    start_date, end_date : str
        ISO date strings for metadata.
    baseline : dict, optional
        Population baseline for divergence. Uses global default if None.

    Returns
    -------
    SpectralProfile
    """
    freqs, coeffs, power = compute_fft(signal)

    h_n = spectral_entropy(freqs, power)
    c_s = spectral_centroid(freqs, power)
    hnr = harmonic_to_noise_ratio(freqs, power)
    beta = spectral_slope(freqs, power)

    timescale = 1.0 / c_s if c_s > 0 else float("inf")
    timescale_label = (
        _format_period(timescale) if np.isfinite(timescale) else "indeterminate"
    )

    periods = find_dominant_periods(freqs, power)
    div = compute_divergence(h_n, c_s, hnr, beta, baseline)

    ac_mask = freqs > 0

    return SpectralProfile(
        spectral_entropy=round(h_n, 4),
        spectral_centroid_hz=round(c_s, 6),
        governing_timescale_hours=(
            round(timescale, 1) if np.isfinite(timescale) else float("inf")
        ),
        governing_timescale_label=timescale_label,
        harmonic_to_noise_ratio_dB=round(hnr, 2),
        spectral_slope=round(beta, 3),
        dominant_periods=periods,
        divergence=div,
        power_spectrum={
            "frequencies": freqs[ac_mask].tolist(),
            "power": power[ac_mask].tolist(),
        },
        metadata={
            "total_events": int(signal.sum()),
            "observation_days": round(len(signal) / 24.0, 1),
            "start_date": start_date,
            "end_date": end_date,
        },
    )


# ---------------------------------------------------------------------------
# Population baseline calibration
# ---------------------------------------------------------------------------

def compute_population_baseline(
    signals: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute baseline statistics from a population of signals.

    Run this on synthetic or real data to calibrate POPULATION_BASELINE.
    """
    metrics = {"entropy": [], "centroid": [], "hnr": [], "slope": []}

    for signal in signals.values():
        freqs, _, power = compute_fft(signal)
        metrics["entropy"].append(spectral_entropy(freqs, power))
        metrics["centroid"].append(spectral_centroid(freqs, power))
        metrics["hnr"].append(harmonic_to_noise_ratio(freqs, power))
        metrics["slope"].append(spectral_slope(freqs, power))

    return {
        "spectral_entropy": {
            "mean": round(float(np.mean(metrics["entropy"])), 4),
            "std": round(float(np.std(metrics["entropy"])), 4),
        },
        "spectral_centroid_hz": {
            "mean": round(float(np.mean(metrics["centroid"])), 6),
            "std": round(float(np.std(metrics["centroid"])), 6),
        },
        "hnr_dB": {
            "mean": round(float(np.mean(metrics["hnr"])), 2),
            "std": round(float(np.std(metrics["hnr"])), 2),
        },
        "spectral_slope": {
            "mean": round(float(np.mean(metrics["slope"])), 4),
            "std": round(float(np.std(metrics["slope"])), 4),
        },
    }


if __name__ == "__main__":
    from ingestion import generate_synthetic_data

    signals = generate_synthetic_data(n_developers=100, days=180)
    baseline = compute_population_baseline(signals)
    print("Calibrated POPULATION_BASELINE from 100 synthetic developers:")
    for key, stats in baseline.items():
        print(f"  {key}: mean={stats['mean']}, std={stats['std']}")
