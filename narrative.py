"""
Fourier Psychology — Narrative Interpretation Engine
====================================================
Deterministic, template-based text generation for spectral profile metrics.
Each function maps metric values to psychologically meaningful interpretations.

All text is structured in tiers — the same metric value always produces the
same interpretation.  No LLM, no randomness.
"""

from __future__ import annotations

from engine import SpectralProfile


# ---------------------------------------------------------------------------
# Individual metric interpretations
# ---------------------------------------------------------------------------

def interpret_entropy(h_n: float) -> str:
    """Interpret normalised spectral entropy Hn in [0, 1]."""
    if h_n < 0.3:
        return (
            "Your behaviour is highly ordered \u2014 dominated by a few strong "
            "repeating cycles. Your actions are primarily governed by internal "
            "routine rather than external circumstances."
        )
    elif h_n < 0.5:
        return (
            "Your behaviour is moderately structured. You have recognisable "
            "patterns but they compete with environmental noise and "
            "circumstantial variation."
        )
    elif h_n < 0.7:
        return (
            "Your behaviour has limited structure. No single pattern "
            "dominates \u2014 your activity is driven by a mix of habit and "
            "external triggers."
        )
    else:
        return (
            "Your behaviour has high entropy \u2014 it doesn\u2019t follow a "
            "dominant repeating pattern. Your activity is responsive to "
            "circumstances rather than governed by routine."
        )


def interpret_centroid(timescale_hours: float) -> str:
    """Interpret governing timescale (1/centroid) in hours."""
    if timescale_hours < 18:
        return (
            "Your behaviour is governed by sub-daily cycles \u2014 you operate "
            "in short, rapid oscillations within each day."
        )
    elif timescale_hours <= 30:
        return (
            "Your governing timescale is approximately daily. The 24-hour "
            "cycle is the strongest organising force in your behaviour. "
            "You respond well to daily routines and daily planning."
        )
    elif timescale_hours <= 120:
        days = timescale_hours / 24
        return (
            f"Your governing timescale is multi-day \u2014 approximately "
            f"{days:.1f} days. The conventional daily schedule doesn\u2019t "
            f"match your natural rhythm. You operate in arcs that span "
            f"multiple days."
        )
    elif timescale_hours <= 200:
        return (
            "Your governing timescale is approximately weekly. Your "
            "behaviour organises itself in weekly arcs rather than daily "
            "routines."
        )
    else:
        return (
            "Your governing timescale is longer than a week. Your behaviour "
            "organises at a resolution that conventional calendars don\u2019t "
            "accommodate."
        )


def interpret_hnr(hnr_dB: float) -> str:
    """Interpret harmonic-to-noise ratio in decibels."""
    if hnr_dB > 10:
        return (
            "Your behaviour is predominantly structured pattern. Internal "
            "rhythms clearly dominate over environmental noise. You are "
            "highly self-governed."
        )
    elif hnr_dB > 5:
        return (
            "Your behaviour has clear patterns that are moderately stronger "
            "than background noise. You have internal structure but it can "
            "be disrupted."
        )
    elif hnr_dB > 2:
        return (
            "Your patterns and noise are roughly balanced. Your internal "
            "structure is present but fragile \u2014 easily overwhelmed by "
            "external demands."
        )
    else:
        return (
            "Noise dominates pattern in your behaviour. Your activity is "
            "primarily reactive to external events rather than driven by "
            "internal rhythm."
        )


def interpret_slope(beta: float) -> str:
    """Interpret spectral slope beta."""
    if beta > 1.5:
        return (
            "Your behaviour is dominated by slow, stable patterns. You "
            "resist rapid change. This indicates strong inertia \u2014 "
            "consistency, but potentially rigidity."
        )
    elif beta >= 0.8:
        return (
            "Your spectral slope is near 1 \u2014 the adaptive zone. Your "
            "behaviour balances stability with responsiveness. This is "
            "characteristic of healthy, flexible self-regulation."
        )
    elif beta >= 0.5:
        return (
            "Your behaviour is tilted toward short-term reactivity. You "
            "respond quickly to new inputs but at the cost of long-term "
            "stability."
        )
    else:
        return (
            "Your behaviour is dominated by rapid, short-term fluctuation. "
            "Long-term patterns are weak. You are highly reactive to "
            "immediate circumstances."
        )


def interpret_divergence(divergence: dict) -> str:
    """Interpret composite divergence score D."""
    D = divergence["composite_D"]
    if D < 1.0:
        return (
            "Your behavioural structure is within the typical range. "
            "Your working pattern is broadly conventional."
        )
    elif D < 1.5:
        return (
            "Your behavioural structure shows moderate divergence from the "
            "population average. Some dimensions of how you work differ "
            "noticeably from the norm."
        )
    elif D < 2.0:
        return (
            "Your behavioural structure is notably different from the "
            "population average. The way you work diverges significantly "
            "from conventional patterns."
        )
    else:
        return (
            "Your behavioural structure is highly atypical \u2014 more than "
            "2 standard deviations from the population average. Your working "
            "pattern is fundamentally different from the assumed norm. This "
            "divergence is measurable and mathematically significant."
        )


# ---------------------------------------------------------------------------
# Composite narrative
# ---------------------------------------------------------------------------

def _entropy_adjective(h_n: float) -> str:
    if h_n < 0.3:
        return "highly ordered"
    elif h_n < 0.5:
        return "moderately structured"
    elif h_n < 0.7:
        return "loosely structured"
    return "largely unstructured"


def _hnr_clause(hnr_dB: float) -> str:
    if hnr_dB > 10:
        return "strong internal rhythms clearly dominate your activity"
    elif hnr_dB > 5:
        return "recognisable patterns emerge above the noise"
    elif hnr_dB > 2:
        return "your patterns and environmental noise are roughly matched"
    return (
        "external events and noise drive more of your activity than "
        "internal rhythm"
    )


def _slope_clause(beta: float) -> str:
    if beta > 1.5:
        return (
            "and your dynamics are slow and inertial \u2014 you resist "
            "rapid change"
        )
    elif beta >= 0.8:
        return (
            "and your dynamics sit in the adaptive zone \u2014 balancing "
            "stability with responsiveness"
        )
    elif beta >= 0.5:
        return (
            "and your dynamics lean toward the reactive \u2014 you respond "
            "quickly but at the cost of long-term stability"
        )
    return (
        "and your dynamics are dominated by short-term fluctuation rather "
        "than sustained rhythm"
    )


def composite_narrative(profile: SpectralProfile) -> str:
    """Synthesise all four metrics into a single coherent paragraph."""
    h_n = profile.spectral_entropy
    timescale = profile.governing_timescale_hours
    hnr = profile.harmonic_to_noise_ratio_dB
    beta = profile.spectral_slope
    D = profile.divergence["composite_D"]

    if timescale < 48:
        time_desc = f"approximately {timescale:.0f} hours"
    else:
        time_desc = f"approximately {timescale / 24:.1f} days"

    narrative = (
        f"Your behavioural spectrum reveals a {_entropy_adjective(h_n)} "
        f"pattern with a governing timescale of {time_desc}. "
        f"{_hnr_clause(hnr).capitalize()}, {_slope_clause(beta)}."
    )

    if D >= 2.0:
        narrative += (
            f" At {D:.1f}\u03c3 from the population mean, your overall "
            f"behavioural structure is highly atypical."
        )
    elif D >= 1.5:
        narrative += (
            f" At {D:.1f}\u03c3 from the population mean, your behavioural "
            f"structure differs notably from the norm."
        )
    elif D >= 1.0:
        narrative += (
            f" At {D:.1f}\u03c3 from the mean, your pattern shows moderate "
            f"divergence from the population."
        )

    return narrative


# ---------------------------------------------------------------------------
# Urgency-driven attention note
# ---------------------------------------------------------------------------

def urgency_attention_note(profile: SpectralProfile) -> str | None:
    """Return advisory text if the profile is consistent with
    urgency-driven attention regulation, otherwise None.
    """
    if (
        profile.divergence["composite_D"] >= 2.0
        and profile.spectral_entropy >= 0.7
        and profile.governing_timescale_hours >= 120
        and profile.harmonic_to_noise_ratio_dB < 2
        and profile.spectral_slope < 0.5
    ):
        return (
            "The pattern of your divergence \u2014 high entropy, long "
            "governing timescale, low pattern-to-noise ratio, and reactive "
            "spectral slope \u2014 is consistent with urgency-driven "
            "attention regulation. If this description resonates with your "
            "experience, it may be worth exploring with a professional. "
            "This is not a diagnosis \u2014 it is a quantitative measurement "
            "of your behavioural structure."
        )
    return None


# ---------------------------------------------------------------------------
# Full interpretation bundle
# ---------------------------------------------------------------------------

def generate_interpretation(profile: SpectralProfile) -> dict:
    """Generate all interpretation text for a spectral profile.

    Returns a dict with keys:
        entropy, centroid, hnr, slope, divergence, composite, urgency_note
    """
    return {
        "entropy": interpret_entropy(profile.spectral_entropy),
        "centroid": interpret_centroid(profile.governing_timescale_hours),
        "hnr": interpret_hnr(profile.harmonic_to_noise_ratio_dB),
        "slope": interpret_slope(profile.spectral_slope),
        "divergence": interpret_divergence(profile.divergence),
        "composite": composite_narrative(profile),
        "urgency_note": urgency_attention_note(profile),
    }
