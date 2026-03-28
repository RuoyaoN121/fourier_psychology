"""
Spectral Sabermetrics — Part 2: Mathematical Processing Engine
===============================================================
Applies FFT-based signal analysis to hourly developer commit time-series
and computes two custom spectral metrics:

* **Cognitive Volatility Score (CVS)** — ratio of high-frequency spectral
  energy (periods < 12 h) to total spectral energy.
* **Flow State Resonance (FSR)** — amplitudes of the 24-hour (circadian)
  and 7-day (weekly) fundamental components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core FFT computation
# ---------------------------------------------------------------------------

# Sample spacing: 1 hour expressed in units of *days* (1/24)
SAMPLE_SPACING_DAYS: float = 1.0 / 24.0


def compute_fft(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run real-valued FFT on an hourly time-series.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of hourly commit counts (real-valued).

    Returns
    -------
    freqs : np.ndarray
        Frequency bins in cycles-per-day.
    coeffs : np.ndarray
        Complex Fourier coefficients from ``rfft``.
    power : np.ndarray
        Power spectrum (|coeffs|²).
    """
    n = len(signal)
    coeffs = rfft(signal)
    freqs = rfftfreq(n, d=SAMPLE_SPACING_DAYS)  # cycles per day
    power = np.abs(coeffs) ** 2
    return freqs, coeffs, power


# ---------------------------------------------------------------------------
# Metric: Cognitive Volatility Score (CVS)
# ---------------------------------------------------------------------------

def cognitive_volatility_score(
    freqs: np.ndarray,
    power: np.ndarray,
    period_threshold_hours: float = 12.0,
) -> float:
    """Fraction of spectral energy in high-frequency bands.

    High frequency is defined as any component whose period is shorter than
    ``period_threshold_hours`` (default 12 h), i.e. frequency >
    24 / period_threshold_hours cycles/day = 2 cycles/day.

    The DC component (frequency = 0) is excluded from both numerator and
    denominator since it represents the mean signal level, not a rhythm.

    Returns a value in [0, 1]; higher means more erratic commit patterns.
    """
    freq_threshold = 24.0 / period_threshold_hours  # cycles per day
    # Exclude DC (freq == 0) — the mean level is not a behavioural rhythm
    ac_mask = freqs > 0
    total_energy = np.sum(power[ac_mask])
    if total_energy == 0:
        return 0.0
    high_freq_energy = np.sum(power[freqs > freq_threshold])
    return float(high_freq_energy / total_energy)


# ---------------------------------------------------------------------------
# Metric: Flow State Resonance (FSR)
# ---------------------------------------------------------------------------

@dataclass
class FlowStateResonance:
    """Container for the two fundamental amplitudes."""
    daily_amplitude: float    # 24-hour / circadian component
    weekly_amplitude: float   # 7-day / weekly component


def _nearest_index(freqs: np.ndarray, target_freq: float) -> int:
    """Return the index of the frequency bin closest to *target_freq*."""
    return int(np.argmin(np.abs(freqs - target_freq)))


def flow_state_resonance(
    freqs: np.ndarray,
    coeffs: np.ndarray,
) -> FlowStateResonance:
    """Extract amplitudes at the 24-hour and 7-day fundamentals.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins (cycles per day).
    coeffs : np.ndarray
        Complex Fourier coefficients.

    Returns
    -------
    FlowStateResonance
        Amplitudes of the daily and weekly components.
    """
    daily_freq = 1.0          # 1 cycle / day  → 24-hour period
    weekly_freq = 1.0 / 7.0   # 1/7 cycle / day → 7-day period

    daily_idx = _nearest_index(freqs, daily_freq)
    weekly_idx = _nearest_index(freqs, weekly_freq)

    daily_amp = float(np.abs(coeffs[daily_idx]))
    weekly_amp = float(np.abs(coeffs[weekly_idx]))

    return FlowStateResonance(daily_amplitude=daily_amp,
                              weekly_amplitude=weekly_amp)


# ---------------------------------------------------------------------------
# Feature matrix builder (for clustering)
# ---------------------------------------------------------------------------

# Number of dominant frequency bins to keep for clustering features.
N_DOMINANT: int = 10


def _top_k_indices(power: np.ndarray, k: int) -> np.ndarray:
    """Indices of the *k* largest values in *power* (excluding DC at idx 0)."""
    power_no_dc = power.copy()
    power_no_dc[0] = 0.0  # ignore DC component
    return np.argsort(power_no_dc)[-k:][::-1]


def build_feature_matrix(
    developer_signals: Dict[str, np.ndarray],
    n_features: int = N_DOMINANT,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a feature matrix from dominant Fourier coefficients.

    For each developer the feature vector is:
        [mag_1, mag_2, …, mag_k, phase_1, phase_2, …, phase_k, CVS, FSR_daily, FSR_weekly]

    Parameters
    ----------
    developer_signals : dict[str, np.ndarray]
        Mapping of developer name → hourly signal.
    n_features : int
        Number of dominant frequency bins to include.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_developers, 2*n_features + 3)``.
    names : list[str]
        Developer names in the same order as rows of *X*.
    feature_names : list[str]
        Human-readable names for each column (for biplot labelling).
    """
    names: List[str] = []
    rows: List[np.ndarray] = []

    for dev, signal in developer_signals.items():
        freqs, coeffs, power = compute_fft(signal)
        top_idx = _top_k_indices(power, n_features)

        magnitudes = np.abs(coeffs[top_idx])
        phases = np.angle(coeffs[top_idx])

        cvs = cognitive_volatility_score(freqs, power)
        fsr = flow_state_resonance(freqs, coeffs)

        feature_vec = np.concatenate([
            magnitudes, phases,               # 2 * n_features
            [cvs, fsr.daily_amplitude, fsr.weekly_amplitude],
        ])
        rows.append(feature_vec)
        names.append(dev)

    X = np.vstack(rows)

    # Standardise features for clustering
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    X = (X - mean) / std

    # Human-readable feature labels for biplot axes
    feature_names: List[str] = (
        [f"Mag ω{i+1}" for i in range(n_features)]
        + [f"Phase ω{i+1}" for i in range(n_features)]
        + ["Cognitive Volatility", "Flow State (Daily)", "Flow State (Weekly)"]
    )

    logger.info("Feature matrix shape: %s", X.shape)
    return X, names, feature_names


# ---------------------------------------------------------------------------
# Convenience: analyse a single developer
# ---------------------------------------------------------------------------

@dataclass
class DeveloperSpectralProfile:
    """Complete spectral analysis result for one developer."""
    name: str
    signal: np.ndarray
    freqs: np.ndarray
    coeffs: np.ndarray
    power: np.ndarray
    cvs: float
    fsr: FlowStateResonance


def analyse_developer(name: str, signal: np.ndarray) -> DeveloperSpectralProfile:
    """Run full spectral analysis on a single developer's signal."""
    freqs, coeffs, power = compute_fft(signal)
    cvs = cognitive_volatility_score(freqs, power)
    fsr = flow_state_resonance(freqs, coeffs)
    return DeveloperSpectralProfile(
        name=name, signal=signal, freqs=freqs,
        coeffs=coeffs, power=power, cvs=cvs, fsr=fsr,
    )


# ---------------------------------------------------------------------------
# Temporal behavioural metrics (for radar charts)
# ---------------------------------------------------------------------------

def commit_volume(signal: np.ndarray) -> float:
    """Total commits across the entire signal."""
    return float(signal.sum())


def weekend_activity_ratio(signal: np.ndarray) -> float:
    """Fraction of total commits that fall on weekends (Sat + Sun).

    Assumes the signal starts at hour 0 of a Monday (the synthetic generator
    does this).  Each week has 168 hours; Saturday = hours 120–143,
    Sunday = hours 144–167.
    """
    total = signal.sum()
    if total == 0:
        return 0.0
    n = len(signal)
    weekend_mask = np.zeros(n, dtype=bool)
    for h in range(n):
        hour_of_week = h % 168
        if hour_of_week >= 120:  # Saturday 00:00 onwards
            weekend_mask[h] = True
    return float(signal[weekend_mask].sum() / total)


def night_owl_index(signal: np.ndarray) -> float:
    """Fraction of total commits made during 'night' hours (22:00–06:00).

    Higher values indicate a developer who works predominantly at night.
    """
    total = signal.sum()
    if total == 0:
        return 0.0
    n = len(signal)
    night_mask = np.zeros(n, dtype=bool)
    for h in range(n):
        hour_of_day = h % 24
        if hour_of_day >= 22 or hour_of_day < 6:  # 22–06 window
            night_mask[h] = True
    return float(signal[night_mask].sum() / total)


def build_radar_profile(name: str, signal: np.ndarray) -> Dict[str, float]:
    """Build a dict of 5 behavioural metrics for radar-chart display.

    Metrics
    -------
    Flow State Resonance : normalised daily FSR amplitude
    Cognitive Volatility : CVS (already 0-1)
    Commit Volume        : total commits (raw — normalised at plot time)
    Weekend Activity     : fraction of commits on weekends (0-1)
    Night-Owl Index      : fraction of commits at night (0-1)
    """
    prof = analyse_developer(name, signal)
    return {
        "Flow State Resonance": prof.fsr.daily_amplitude,
        "Cognitive Volatility": prof.cvs,
        "Commit Volume": commit_volume(signal),
        "Weekend Activity": weekend_activity_ratio(signal),
        "Night-Owl Index": night_owl_index(signal),
    }


def rank_developers_by_cvs(
    signals: Dict[str, np.ndarray],
) -> List[Tuple[str, float]]:
    """Return a list of ``(name, cvs)`` sorted ascending by CVS."""
    results = []
    for dev, sig in signals.items():
        freqs, coeffs, power = compute_fft(sig)
        cvs = cognitive_volatility_score(freqs, power)
        results.append((dev, cvs))
    results.sort(key=lambda t: t[1])
    return results


# ---------------------------------------------------------------------------
# Macro-Sabermetrics Engine
# ---------------------------------------------------------------------------

def compute_rolling_macro_volatility(
    series: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Calculate the rolling Organizational Volatility Score.

    Applies a rolling window of specified length (days) over the daily push
    signal. For each window, computes the rfft and sums the high-frequency
    spectral energy (excluding DC and the lowest frequencies).

    Parameters
    ----------
    series : pd.Series
        Daily push counts.
    window : int
        Rolling window size in days (default: 30).

    Returns
    -------
    pd.Series
        Rolling volatility score, with initial `window - 1` days as NaNs.
    """
    out = pd.Series(index=series.index, dtype=float)
    
    # We want to measure "high frequency noise" in a 30-day window
    # Frequencies will range from 0 (DC) to 0.5 cycles/day (Nyquist)
    # 0.5 cycles/day = 2-day period.
    # Let's define "high-frequency" as periods < 7 days (i.e. freq > 1/7 = 0.14)
    # to capture chaotic short-term variations vs steady weekly/monthly rhythms.
    freq_threshold = 1.0 / 7.0

    values = series.values
    for i in range(window - 1, len(values)):
        window_slice = values[i - window + 1 : i + 1]
        
        # Only process if we have valid numeric data
        if np.isnan(window_slice).any():
            continue
            
        # De-mean to avoid huge DC spike dominating the energy sum
        window_slice = window_slice - np.mean(window_slice)
        
        # Apply Hanning window to reduce spectral leakage at edges
        windowed = window_slice * np.hanning(window)

        coeffs = rfft(windowed)
        freqs = rfftfreq(window, d=1.0)  # d=1 day
        power = np.abs(coeffs) ** 2
        
        # Sum energy in high-frequency bands
        high_freq_mask = freqs > freq_threshold
        high_freq_energy = power[high_freq_mask].sum()
        
        # Optionally, normalise by total AC energy to get a ratio,
        # but the prompt asks to "sum the spectral energy". We will just use the sum,
        # perhaps scaled for readability.
        out.iloc[i] = high_freq_energy

    # Smooth the resulting score to make the chart trend clearer
    # using a simple 7-day moving average on the volatility score itself
    out = out.rolling(7, min_periods=1).mean()
    
    return out

