"""
Fourier Psychology — Tests for the Narrative Interpretation Engine
==================================================================
Verifies that each interpretation function produces the correct tier
of text for known metric values, and that the composite narrative
and urgency note trigger appropriately.

Run:
    python -m pytest test_narrative.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from engine import SpectralProfile
from narrative import (
    interpret_entropy,
    interpret_centroid,
    interpret_hnr,
    interpret_slope,
    interpret_divergence,
    composite_narrative,
    urgency_attention_note,
    generate_interpretation,
)


# ---------------------------------------------------------------------------
# Helper: build a SpectralProfile with controllable values
# ---------------------------------------------------------------------------

def _make_profile(
    entropy=0.5,
    centroid_hz=0.04,
    timescale=25.0,
    hnr=6.0,
    slope=0.9,
    D=1.0,
    z_entropy=0.0,
    z_centroid=0.0,
    z_hnr=0.0,
    z_slope=0.0,
) -> SpectralProfile:
    return SpectralProfile(
        spectral_entropy=entropy,
        spectral_centroid_hz=centroid_hz,
        governing_timescale_hours=timescale,
        governing_timescale_label=f"~{timescale:.0f} hours",
        harmonic_to_noise_ratio_dB=hnr,
        spectral_slope=slope,
        dominant_periods=[],
        divergence={
            "z_entropy": z_entropy,
            "z_centroid": z_centroid,
            "z_hnr": z_hnr,
            "z_slope": z_slope,
            "composite_D": D,
        },
        power_spectrum={"frequencies": [], "power": []},
        metadata={
            "total_events": 1000,
            "observation_days": 90.0,
            "start_date": "",
            "end_date": "",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Entropy tiers
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretEntropy:
    def test_tier_highly_ordered(self):
        text = interpret_entropy(0.15)
        assert "highly ordered" in text

    def test_tier_moderately_structured(self):
        text = interpret_entropy(0.4)
        assert "moderately structured" in text

    def test_tier_limited_structure(self):
        text = interpret_entropy(0.6)
        assert "limited structure" in text

    def test_tier_high_entropy(self):
        text = interpret_entropy(0.85)
        assert "high entropy" in text

    def test_boundary_030(self):
        assert "moderately" in interpret_entropy(0.3)

    def test_boundary_050(self):
        assert "limited structure" in interpret_entropy(0.5)

    def test_boundary_070(self):
        assert "high entropy" in interpret_entropy(0.7)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Centroid / timescale tiers
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretCentroid:
    def test_sub_daily(self):
        text = interpret_centroid(10.0)
        assert "sub-daily" in text

    def test_daily(self):
        text = interpret_centroid(24.0)
        assert "daily" in text.lower()

    def test_multi_day(self):
        text = interpret_centroid(72.0)
        assert "multi-day" in text

    def test_weekly(self):
        text = interpret_centroid(160.0)
        assert "weekly" in text.lower()

    def test_longer_than_week(self):
        text = interpret_centroid(250.0)
        assert "longer than a week" in text


# ═══════════════════════════════════════════════════════════════════════════
# Tests: HNR tiers
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretHNR:
    def test_predominantly_structured(self):
        text = interpret_hnr(15.0)
        assert "predominantly structured" in text

    def test_clear_patterns(self):
        text = interpret_hnr(7.0)
        assert "clear patterns" in text

    def test_balanced(self):
        text = interpret_hnr(3.5)
        assert "roughly balanced" in text

    def test_noise_dominates(self):
        text = interpret_hnr(0.5)
        assert "Noise dominates" in text


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Slope tiers
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretSlope:
    def test_rigid(self):
        text = interpret_slope(1.8)
        assert "slow, stable" in text

    def test_adaptive(self):
        text = interpret_slope(1.0)
        assert "adaptive zone" in text

    def test_reactive(self):
        text = interpret_slope(0.6)
        assert "short-term reactivity" in text

    def test_chaotic(self):
        text = interpret_slope(0.3)
        assert "short-term fluctuation" in text


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Divergence tiers
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretDivergence:
    def test_typical(self):
        text = interpret_divergence({"composite_D": 0.5})
        assert "typical range" in text

    def test_moderate(self):
        text = interpret_divergence({"composite_D": 1.2})
        assert "moderate divergence" in text

    def test_notable(self):
        text = interpret_divergence({"composite_D": 1.7})
        assert "notably different" in text

    def test_highly_atypical(self):
        text = interpret_divergence({"composite_D": 2.5})
        assert "highly atypical" in text


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Composite narrative
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositeNarrative:
    def test_returns_nonempty_string(self):
        profile = _make_profile()
        text = composite_narrative(profile)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_includes_timescale(self):
        profile = _make_profile(timescale=24.0)
        text = composite_narrative(profile)
        assert "24 hours" in text

    def test_high_divergence_mentioned(self):
        profile = _make_profile(D=2.5)
        text = composite_narrative(profile)
        assert "2.5" in text
        assert "atypical" in text

    def test_low_divergence_no_extra_sentence(self):
        profile = _make_profile(D=0.5)
        text = composite_narrative(profile)
        assert "atypical" not in text


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Urgency attention note
# ═══════════════════════════════════════════════════════════════════════════

class TestUrgencyNote:
    def test_triggers_when_all_conditions_met(self):
        profile = _make_profile(
            entropy=0.85, timescale=150.0, hnr=0.5, slope=0.3, D=2.5,
        )
        note = urgency_attention_note(profile)
        assert note is not None
        assert "urgency-driven" in note
        assert "not a diagnosis" in note

    def test_does_not_trigger_when_D_low(self):
        profile = _make_profile(
            entropy=0.85, timescale=150.0, hnr=0.5, slope=0.3, D=1.5,
        )
        assert urgency_attention_note(profile) is None

    def test_does_not_trigger_when_entropy_low(self):
        profile = _make_profile(
            entropy=0.4, timescale=150.0, hnr=0.5, slope=0.3, D=2.5,
        )
        assert urgency_attention_note(profile) is None

    def test_does_not_trigger_when_hnr_high(self):
        profile = _make_profile(
            entropy=0.85, timescale=150.0, hnr=8.0, slope=0.3, D=2.5,
        )
        assert urgency_attention_note(profile) is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Full bundle
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateInterpretation:
    def test_returns_all_keys(self):
        profile = _make_profile()
        interp = generate_interpretation(profile)
        for key in [
            "entropy", "centroid", "hnr", "slope",
            "divergence", "composite", "urgency_note",
        ]:
            assert key in interp

    def test_all_values_are_strings_or_none(self):
        profile = _make_profile()
        interp = generate_interpretation(profile)
        for key, val in interp.items():
            assert isinstance(val, (str, type(None))), f"{key} is {type(val)}"
