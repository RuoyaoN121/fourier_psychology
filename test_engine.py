"""
Fourier Psychology — Unit Tests for the Spectral Analysis Engine
================================================================
Tests the four spectral metrics using signals with known mathematical
properties:

* Pure sine waves → low entropy, precise centroid, high HNR
* White noise → high entropy, low HNR, β ≈ 0
* Coloured noise → specific spectral slopes (β ≈ 1 pink, β ≈ 2 brown)
* Two-tone composites → correct peak detection

Run:
    python -m pytest test_engine.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import rfft, rfftfreq, irfft

from engine import (
    SAMPLE_SPACING_HOURS,
    compute_fft,
    spectral_entropy,
    spectral_centroid,
    harmonic_to_noise_ratio,
    spectral_slope,
    find_dominant_periods,
    compute_divergence,
    analyse,
    compute_population_baseline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAYS = 90
HOURS = DAYS * 24
T = np.arange(HOURS, dtype=np.float64)


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _pure_sine(period_hours: float, amplitude: float = 4.0, offset: float = 5.0):
    """Pure sine wave with given period."""
    return offset + amplitude * np.sin(2 * np.pi * T / period_hours)


def _white_noise(seed: int = 42):
    """Uniform random noise — no dominant frequency."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 10, size=HOURS).astype(np.float64)


def _colored_noise(beta: float, seed: int = 42):
    """Generate noise with power spectrum P(f) ∝ 1/f^β."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(HOURS)
    coeffs = rfft(white)
    freqs = rfftfreq(HOURS, d=SAMPLE_SPACING_HOURS)
    freqs_safe = freqs.copy()
    freqs_safe[0] = 1.0  # avoid division by zero at DC
    scaling = freqs_safe ** (-beta / 2.0)
    scaling[0] = 0  # zero DC
    return irfft(coeffs * scaling, n=HOURS)


def _two_tone(periods=(24.0, 168.0)):
    """Sum of two sine waves at different periods."""
    sig = np.full(HOURS, 5.0)
    for p in periods:
        sig = sig + 3.0 * np.sin(2 * np.pi * T / p)
    return sig


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Spectral Entropy
# ═══════════════════════════════════════════════════════════════════════════

class TestSpectralEntropy:
    def test_pure_sine_low_entropy(self):
        """A single dominant frequency → nearly all probability at one bin."""
        signal = _pure_sine(24.0)
        freqs, _, power = compute_fft(signal)
        h = spectral_entropy(freqs, power)
        assert h < 0.15, f"Pure sine entropy should be < 0.15, got {h:.4f}"

    def test_white_noise_high_entropy(self):
        """Flat power spectrum → near-uniform probability → max entropy."""
        signal = _white_noise()
        freqs, _, power = compute_fft(signal)
        h = spectral_entropy(freqs, power)
        assert h > 0.90, f"White noise entropy should be > 0.90, got {h:.4f}"

    def test_two_tone_between_sine_and_noise(self):
        """Two frequencies → more entropy than one, less than noise."""
        f1, _, p1 = compute_fft(_pure_sine(24.0))
        f2, _, p2 = compute_fft(_white_noise())
        f3, _, p3 = compute_fft(_two_tone())

        h_sine = spectral_entropy(f1, p1)
        h_noise = spectral_entropy(f2, p2)
        h_two = spectral_entropy(f3, p3)

        assert h_sine < h_two < h_noise, (
            f"Expected sine ({h_sine:.3f}) < two-tone ({h_two:.3f}) < noise ({h_noise:.3f})"
        )

    def test_range_zero_to_one(self):
        """Entropy must always be in [0, 1]."""
        for signal in [_pure_sine(24.0), _white_noise(), _two_tone()]:
            freqs, _, power = compute_fft(signal)
            h = spectral_entropy(freqs, power)
            assert 0.0 <= h <= 1.0, f"Entropy out of range: {h}"

    def test_zero_signal(self):
        """All-zero signal → no spectral energy → entropy 0."""
        signal = np.zeros(HOURS)
        freqs, _, power = compute_fft(signal)
        assert spectral_entropy(freqs, power) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Spectral Centroid
# ═══════════════════════════════════════════════════════════════════════════

class TestSpectralCentroid:
    def test_24h_sine(self):
        """Pure 24h sine → centroid at 1/24 cycles per hour."""
        signal = _pure_sine(24.0)
        freqs, _, power = compute_fft(signal)
        c = spectral_centroid(freqs, power)
        expected = 1.0 / 24.0
        assert abs(c - expected) < 0.002, (
            f"24h sine centroid should be ~{expected:.5f}, got {c:.5f}"
        )

    def test_12h_sine(self):
        """Pure 12h sine → centroid at 1/12 cycles per hour."""
        signal = _pure_sine(12.0)
        freqs, _, power = compute_fft(signal)
        c = spectral_centroid(freqs, power)
        expected = 1.0 / 12.0
        assert abs(c - expected) < 0.002, (
            f"12h sine centroid should be ~{expected:.5f}, got {c:.5f}"
        )

    def test_governing_timescale_from_centroid(self):
        """1/centroid should recover the dominant period in hours."""
        signal = _pure_sine(24.0)
        freqs, _, power = compute_fft(signal)
        c = spectral_centroid(freqs, power)
        timescale = 1.0 / c
        assert abs(timescale - 24.0) < 1.0, (
            f"Governing timescale should be ~24h, got {timescale:.1f}h"
        )

    def test_higher_freq_sine_has_larger_centroid(self):
        """Shorter period → higher frequency → larger centroid."""
        f1, _, p1 = compute_fft(_pure_sine(24.0))
        f2, _, p2 = compute_fft(_pure_sine(12.0))
        c_24 = spectral_centroid(f1, p1)
        c_12 = spectral_centroid(f2, p2)
        assert c_12 > c_24, "12h sine should have larger centroid than 24h sine"

    def test_zero_signal(self):
        signal = np.zeros(HOURS)
        freqs, _, power = compute_fft(signal)
        assert spectral_centroid(freqs, power) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Harmonic-to-Noise Ratio
# ═══════════════════════════════════════════════════════════════════════════

class TestHarmonicToNoiseRatio:
    def test_pure_sine_high_hnr(self):
        """All spectral energy at one peak → very high HNR."""
        signal = _pure_sine(24.0)
        freqs, _, power = compute_fft(signal)
        hnr = harmonic_to_noise_ratio(freqs, power)
        assert hnr > 10.0, f"Pure sine HNR should be > 10 dB, got {hnr:.1f} dB"

    def test_white_noise_low_hnr(self):
        """Flat spectrum → no real peaks → low or negative HNR."""
        signal = _white_noise()
        freqs, _, power = compute_fft(signal)
        hnr = harmonic_to_noise_ratio(freqs, power)
        assert hnr < 5.0, f"White noise HNR should be < 5 dB, got {hnr:.1f} dB"

    def test_sine_much_higher_than_noise(self):
        """Pure sine HNR should exceed white noise HNR by a wide margin."""
        f1, _, p1 = compute_fft(_pure_sine(24.0))
        f2, _, p2 = compute_fft(_white_noise())
        hnr_sine = harmonic_to_noise_ratio(f1, p1)
        hnr_noise = harmonic_to_noise_ratio(f2, p2)
        assert hnr_sine > hnr_noise + 5, (
            f"Sine HNR ({hnr_sine:.1f}) should exceed noise HNR ({hnr_noise:.1f}) by > 5 dB"
        )

    def test_zero_signal(self):
        signal = np.zeros(HOURS)
        freqs, _, power = compute_fft(signal)
        assert harmonic_to_noise_ratio(freqs, power) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Spectral Slope
# ═══════════════════════════════════════════════════════════════════════════

class TestSpectralSlope:
    def test_white_noise_slope_near_zero(self):
        """White noise has flat spectrum → β ≈ 0."""
        signal = _colored_noise(beta=0.0)
        freqs, _, power = compute_fft(signal)
        beta = spectral_slope(freqs, power)
        assert abs(beta) < 0.5, f"White noise β should be ~0, got {beta:.2f}"

    def test_pink_noise_slope_near_one(self):
        """Pink (1/f) noise → β ≈ 1."""
        signal = _colored_noise(beta=1.0)
        freqs, _, power = compute_fft(signal)
        beta = spectral_slope(freqs, power)
        assert 0.5 < beta < 1.5, f"Pink noise β should be ~1, got {beta:.2f}"

    def test_brown_noise_slope_near_two(self):
        """Brown (1/f²) noise → β ≈ 2."""
        signal = _colored_noise(beta=2.0)
        freqs, _, power = compute_fft(signal)
        beta = spectral_slope(freqs, power)
        assert 1.5 < beta < 2.5, f"Brown noise β should be ~2, got {beta:.2f}"

    def test_ordering_white_pink_brown(self):
        """β should increase monotonically: white < pink < brown."""
        slopes = []
        for target_beta in [0.0, 1.0, 2.0]:
            signal = _colored_noise(beta=target_beta, seed=99)
            freqs, _, power = compute_fft(signal)
            slopes.append(spectral_slope(freqs, power))
        assert slopes[0] < slopes[1] < slopes[2], (
            f"Expected monotonic ordering, got {slopes}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Dominant Periods
# ═══════════════════════════════════════════════════════════════════════════

class TestDominantPeriods:
    def test_24h_sine_finds_24h_peak(self):
        """Should identify the 24-hour cycle as the dominant period."""
        signal = _pure_sine(24.0)
        freqs, _, power = compute_fft(signal)
        periods = find_dominant_periods(freqs, power)
        assert len(periods) > 0
        top_period = periods[0]["period_hours"]
        assert abs(top_period - 24.0) < 1.0, (
            f"Should find ~24h peak, got {top_period}h"
        )

    def test_two_tone_finds_both_peaks(self):
        """Should find both 24h and ~168h peaks in a two-tone signal."""
        signal = _two_tone((24.0, 168.0))
        freqs, _, power = compute_fft(signal)
        periods = find_dominant_periods(freqs, power, n_peaks=5)
        found = [p["period_hours"] for p in periods]
        has_daily = any(abs(h - 24.0) < 2.0 for h in found)
        has_weekly = any(abs(h - 168.0) < 20.0 for h in found)
        assert has_daily, f"Should find ~24h peak in {found}"
        assert has_weekly, f"Should find ~168h peak in {found}"

    def test_periods_have_labels(self):
        """Each period should include a human-readable label."""
        signal = _pure_sine(24.0)
        freqs, _, power = compute_fft(signal)
        periods = find_dominant_periods(freqs, power)
        for p in periods:
            assert "label" in p
            assert isinstance(p["label"], str)
            assert p["label"].startswith("~")

    def test_periods_sorted_by_power(self):
        """Periods should be returned in descending power order."""
        signal = _two_tone()
        freqs, _, power = compute_fft(signal)
        periods = find_dominant_periods(freqs, power)
        powers = [p["power"] for p in periods]
        assert powers == sorted(powers, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Divergence
# ═══════════════════════════════════════════════════════════════════════════

class TestDivergence:
    BASELINE = {
        "spectral_entropy": {"mean": 0.5, "std": 0.1},
        "spectral_centroid_hz": {"mean": 0.04, "std": 0.01},
        "hnr_dB": {"mean": 8.0, "std": 2.0},
        "spectral_slope": {"mean": 1.0, "std": 0.2},
    }

    def test_at_mean_zero_divergence(self):
        """Values exactly at the population mean → D = 0."""
        div = compute_divergence(0.5, 0.04, 8.0, 1.0, self.BASELINE)
        assert div["composite_D"] == 0.0
        assert div["z_entropy"] == 0.0
        assert div["z_centroid"] == 0.0
        assert div["z_hnr"] == 0.0
        assert div["z_slope"] == 0.0

    def test_one_sigma_on_one_metric(self):
        """1σ away on entropy only → D = 1.0."""
        div = compute_divergence(0.6, 0.04, 8.0, 1.0, self.BASELINE)
        assert abs(div["z_entropy"] - 1.0) < 0.01
        assert abs(div["composite_D"] - 1.0) < 0.01

    def test_two_sigma_on_all_metrics(self):
        """2σ away on all four metrics → D = sqrt(4 * 4) = 4.0."""
        div = compute_divergence(0.7, 0.06, 12.0, 1.4, self.BASELINE)
        expected_D = np.sqrt(4 * 2.0**2)  # sqrt(16) = 4.0
        assert abs(div["composite_D"] - expected_D) < 0.01

    def test_z_score_signs(self):
        """Z-scores should be positive when above mean, negative when below."""
        div = compute_divergence(0.6, 0.03, 6.0, 1.2, self.BASELINE)
        assert div["z_entropy"] > 0  # 0.6 > 0.5
        assert div["z_centroid"] < 0  # 0.03 < 0.04
        assert div["z_hnr"] < 0  # 6.0 < 8.0
        assert div["z_slope"] > 0  # 1.2 > 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Full Pipeline (analyse)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyse:
    def test_returns_complete_profile(self):
        """The profile object should have all expected fields."""
        signal = _pure_sine(24.0)
        profile = analyse(signal, start_date="2024-01-01", end_date="2024-03-31")

        assert 0 <= profile.spectral_entropy <= 1
        assert profile.spectral_centroid_hz > 0
        assert profile.governing_timescale_hours > 0
        assert isinstance(profile.governing_timescale_label, str)
        assert isinstance(profile.harmonic_to_noise_ratio_dB, float)
        assert isinstance(profile.spectral_slope, float)
        assert len(profile.dominant_periods) > 0
        assert "composite_D" in profile.divergence
        assert "frequencies" in profile.power_spectrum
        assert "power" in profile.power_spectrum
        assert profile.metadata["total_events"] > 0
        assert profile.metadata["observation_days"] == 90.0
        assert profile.metadata["start_date"] == "2024-01-01"

    def test_sine_vs_noise_contrast(self):
        """Sine and noise profiles should differ in the expected directions."""
        sine_p = analyse(_pure_sine(24.0))
        noise_p = analyse(_white_noise())

        assert sine_p.spectral_entropy < noise_p.spectral_entropy
        assert sine_p.harmonic_to_noise_ratio_dB > noise_p.harmonic_to_noise_ratio_dB

    def test_custom_baseline(self):
        """Passing a custom baseline should affect divergence but not metrics."""
        signal = _pure_sine(24.0)
        baseline = {
            "spectral_entropy": {"mean": 0.5, "std": 0.1},
            "spectral_centroid_hz": {"mean": 0.04, "std": 0.01},
            "hnr_dB": {"mean": 8.0, "std": 2.0},
            "spectral_slope": {"mean": 1.0, "std": 0.2},
        }
        p1 = analyse(signal)
        p2 = analyse(signal, baseline=baseline)

        # Metrics should be identical
        assert p1.spectral_entropy == p2.spectral_entropy
        assert p1.spectral_centroid_hz == p2.spectral_centroid_hz
        # Divergence may differ (different baseline)
        assert isinstance(p2.divergence["composite_D"], float)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Population Baseline Calibration
# ═══════════════════════════════════════════════════════════════════════════

class TestPopulationBaseline:
    def test_compute_from_synthetic_signals(self):
        """Should produce valid baseline stats from a set of signals."""
        from ingestion import generate_synthetic_data

        signals = generate_synthetic_data(n_developers=10, days=90, seed=42)
        baseline = compute_population_baseline(signals)

        for key in ["spectral_entropy", "spectral_centroid_hz", "hnr_dB", "spectral_slope"]:
            assert key in baseline
            assert "mean" in baseline[key]
            assert "std" in baseline[key]
            assert isinstance(baseline[key]["mean"], float)
            assert baseline[key]["std"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: FFT Mechanics
# ═══════════════════════════════════════════════════════════════════════════

class TestFFTMechanics:
    def test_frequency_units_are_cycles_per_hour(self):
        """With d=1 hour, frequency step should be 1/N cycles per hour."""
        signal = np.zeros(HOURS)
        freqs, _, _ = compute_fft(signal)
        expected_step = 1.0 / HOURS
        actual_step = freqs[1] - freqs[0]
        assert abs(actual_step - expected_step) < 1e-10

    def test_parseval_energy_conservation(self):
        """Total power in frequency domain ≈ time-domain energy."""
        rng = np.random.default_rng(77)
        signal = rng.standard_normal(HOURS)
        freqs, coeffs, power = compute_fft(signal)
        time_energy = np.sum(signal**2)
        freq_energy = (
            power[0] + 2 * np.sum(power[1:-1]) + power[-1]
        ) / len(signal)
        assert abs(freq_energy - time_energy) < time_energy * 1e-10, (
            "Parseval's theorem violated — FFT pipeline has a bug."
        )

    def test_dc_excluded_from_ac_mask(self):
        """The AC mask (freqs > 0) should exclude exactly one element (DC)."""
        signal = np.ones(HOURS)
        freqs, _, _ = compute_fft(signal)
        ac_count = np.sum(freqs > 0)
        total_count = len(freqs)
        assert ac_count == total_count - 1
