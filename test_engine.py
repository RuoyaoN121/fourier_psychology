"""
Spectral Sabermetrics — Unit Tests for the FFT Engine
======================================================
Generates two archetypal synthetic developer signals and asserts that the
spectral metrics land in the correct regions:

* **Healthy Developer** — a clean 24-hour sine-wave → high FSR daily
  amplitude, low CVS.
* **Burned-Out Developer** — random high-frequency noise → high CVS,
  negligible FSR daily amplitude.

Run:
    python -m pytest test_engine.py -v
    # or simply:
    python test_engine.py
"""

from __future__ import annotations

import unittest

import numpy as np

from engine import (
    compute_fft,
    cognitive_volatility_score,
    flow_state_resonance,
    SAMPLE_SPACING_DAYS,
)

# ---------------------------------------------------------------------------
# Constants shared by all tests
# ---------------------------------------------------------------------------

DAYS = 90
HOURS = DAYS * 24
T = np.arange(HOURS, dtype=np.float64)  # hourly time axis


class TestHealthyDeveloper(unittest.TestCase):
    """A clean 24-hour sine wave → strong daily rhythm, minimal noise."""

    @classmethod
    def setUpClass(cls):
        # Pure circadian signal: commits peak once every 24 hours.
        cls.signal = 5.0 + 4.0 * np.sin(2 * np.pi * T / 24.0)
        cls.freqs, cls.coeffs, cls.power = compute_fft(cls.signal)
        cls.cvs = cognitive_volatility_score(cls.freqs, cls.power)
        cls.fsr = flow_state_resonance(cls.freqs, cls.coeffs)

    # ── FSR assertions ──

    def test_high_daily_amplitude(self):
        """The 24-hour fundamental should dominate the spectrum."""
        self.assertGreater(
            self.fsr.daily_amplitude, 100.0,
            "Healthy developer must have a strong 24-h FSR amplitude.",
        )

    def test_weekly_amplitude_low_relative_to_daily(self):
        """With no weekly modulation, weekly amp should be much smaller."""
        self.assertLess(
            self.fsr.weekly_amplitude,
            self.fsr.daily_amplitude * 0.1,
            "Without weekly modulation, weekly amplitude should be negligible.",
        )

    # ── CVS assertions ──

    def test_low_cognitive_volatility(self):
        """A pure 24-h signal has virtually no high-freq energy."""
        self.assertLess(
            self.cvs, 0.05,
            "Healthy developer CVS must be near zero (< 0.05).",
        )


class TestBurnedOutDeveloper(unittest.TestCase):
    """Random high-frequency noise → no circadian rhythm, high volatility."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        # Uniform random noise: every hour is independent.
        cls.signal = rng.uniform(0, 10, size=HOURS).astype(np.float64)
        cls.freqs, cls.coeffs, cls.power = compute_fft(cls.signal)
        cls.cvs = cognitive_volatility_score(cls.freqs, cls.power)
        cls.fsr = flow_state_resonance(cls.freqs, cls.coeffs)

    # ── CVS assertions ──

    def test_high_cognitive_volatility(self):
        """Flat white noise distributes energy uniformly — most is high-freq."""
        self.assertGreater(
            self.cvs, 0.8,
            "Burned-out developer CVS must be high (> 0.8).",
        )

    # ── FSR assertions ──

    def test_negligible_daily_amplitude(self):
        """Random noise has no 24-h peak worth noting relative to the signal
        length.  We compare against the healthy developer's expected amplitude
        (~4 * HOURS / 2 ≈ 4320) — the noise peak should be far smaller."""
        # With N=2160 random samples, the expected |coefficient| at any bin
        # is O(sqrt(N) * sigma) ≈ 46 * 2.9 ≈ 134.  The healthy dev's daily
        # amplitude is ~4 * N/2 = 4320.  So the noise peak should be << 500.
        self.assertLess(
            self.fsr.daily_amplitude, 500.0,
            "Burned-out developer should have a negligible daily FSR amplitude.",
        )

    def test_cvs_much_higher_than_healthy(self):
        """Cross-archetype sanity: burned-out CVS >> healthy CVS."""
        healthy_signal = 5.0 + 4.0 * np.sin(2 * np.pi * T / 24.0)
        freqs_h, _, power_h = compute_fft(healthy_signal)
        cvs_healthy = cognitive_volatility_score(freqs_h, power_h)

        self.assertGreater(
            self.cvs - cvs_healthy, 0.5,
            "Burned-out CVS must exceed healthy CVS by a wide margin (> 0.5).",
        )


class TestFFTMechanics(unittest.TestCase):
    """Low-level sanity checks on the FFT pipeline itself."""

    def test_frequency_resolution(self):
        """rfftfreq with d=1/24 should produce cycles-per-day."""
        signal = np.zeros(HOURS)
        freqs, _, _ = compute_fft(signal)
        # The frequency step should be 1 / (HOURS * d) = 24 / HOURS
        expected_step = 24.0 / HOURS
        actual_step = freqs[1] - freqs[0]
        self.assertAlmostEqual(actual_step, expected_step, places=8)

    def test_parseval_energy_conservation(self):
        """Total power in frequency domain ≈ time-domain energy (Parseval)."""
        rng = np.random.default_rng(77)
        signal = rng.standard_normal(HOURS)
        freqs, coeffs, power = compute_fft(signal)
        time_energy = np.sum(signal ** 2)
        # For rfft: sum of |c|^2 counts DC and Nyquist once, others twice.
        freq_energy = (
            power[0]
            + 2 * np.sum(power[1:-1])
            + power[-1]
        ) / len(signal)
        self.assertAlmostEqual(
            freq_energy, time_energy, delta=time_energy * 1e-10,
            msg="Parseval's theorem violated — FFT pipeline has a bug.",
        )


if __name__ == "__main__":
    unittest.main()
