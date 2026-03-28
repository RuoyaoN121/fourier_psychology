"""
Fourier Psychology — Data Ingestion Pipeline
=============================================
Three data sources:

1. **Public GitHub API** — fetches a single user's public events (no auth).
2. **GitHub Archive BigQuery** — bulk extraction (requires GCP credentials).
3. **Synthetic** — generates realistic signals for testing and baselines.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BigQuery extraction
# ---------------------------------------------------------------------------

_BQ_QUERY_TEMPLATE = """
WITH top_devs AS (
    SELECT
        actor.login AS developer,
        COUNT(*) AS push_count
    FROM
        `githubarchive.day.20*`
    WHERE
        _TABLE_SUFFIX BETWEEN
            FORMAT_DATE('%y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY))
            AND FORMAT_DATE('%y%m%d', CURRENT_DATE())
        AND type = 'PushEvent'
        AND actor.login IS NOT NULL
    GROUP BY
        developer
    ORDER BY
        push_count DESC
    LIMIT {n}
)
SELECT
    actor.login AS developer,
    created_at
FROM
    `githubarchive.day.20*`
INNER JOIN
    top_devs ON actor.login = top_devs.developer
WHERE
    _TABLE_SUFFIX BETWEEN
        FORMAT_DATE('%y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY))
        AND FORMAT_DATE('%y%m%d', CURRENT_DATE())
    AND type = 'PushEvent'
"""


def fetch_from_bigquery(n: int = 100, days: int = 180) -> pd.DataFrame:
    """Run the extraction query against BigQuery and return a raw DataFrame.

    Returns a DataFrame with columns ``['developer', 'created_at']``.
    Requires a configured Google Cloud project (``GOOGLE_CLOUD_PROJECT`` env
    var or application-default credentials).
    """
    from google.cloud import bigquery  # lazy — optional dependency

    client = bigquery.Client()
    query = _BQ_QUERY_TEMPLATE.format(n=n, days=days)
    logger.info("Submitting BigQuery job (top %d devs, %d-day window)…", n, days)
    df = client.query(query).to_dataframe()
    logger.info("BigQuery returned %d rows for %d developers.",
                len(df), df["developer"].nunique())
    return df


# ---------------------------------------------------------------------------
# Hourly time-series builder
# ---------------------------------------------------------------------------

def build_hourly_timeseries(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Aggregate raw events into zero-filled hourly commit-count arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``developer`` (str) and ``created_at`` (datetime-like)
        columns.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from developer login to a 1-D array of hourly commit counts,
        uniformly sampled and zero-filled so it is ready for FFT.
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["hour"] = df["created_at"].dt.floor("h")

    # Global time range shared by every developer
    hour_min = df["hour"].min()
    hour_max = df["hour"].max()
    full_range = pd.date_range(start=hour_min, end=hour_max, freq="h")

    signals: Dict[str, np.ndarray] = {}
    for dev, grp in df.groupby("developer"):
        counts = grp.groupby("hour").size()
        counts = counts.reindex(full_range, fill_value=0)
        signals[str(dev)] = counts.values.astype(np.float64)

    logger.info("Built hourly signals for %d developers (%d hours each).",
                len(signals), len(full_range))
    return signals


# ---------------------------------------------------------------------------
# Demo / synthetic data fallback
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_developers: int = 100,
    days: int = 180,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create realistic synthetic hourly commit signals.

    Each developer gets:
    * A circadian (24-h) rhythm with a random phase offset.
    * A weekly (168-h) rhythm (lower activity on weekends).
    * Poisson-distributed noise on top.

    This allows the full pipeline to be demonstrated without BigQuery.
    """
    rng = np.random.default_rng(seed)
    hours = days * 24
    t = np.arange(hours, dtype=np.float64)

    # Pre-generate some plausible developer "personality" names
    fake_names = [f"dev_{i:03d}" for i in range(n_developers)]

    signals: Dict[str, np.ndarray] = {}
    for name in fake_names:
        # Random circadian phase (peak hour of day)
        phase_24 = rng.uniform(0, 2 * np.pi)
        # Random weekly phase
        phase_168 = rng.uniform(0, 2 * np.pi)
        # Random amplitudes
        amp_24 = rng.uniform(1.5, 5.0)
        amp_168 = rng.uniform(0.5, 2.0)
        # Base rate
        base = rng.uniform(0.3, 1.5)

        signal = (
            base
            + amp_24 * np.sin(2 * np.pi * t / 24 + phase_24)
            + amp_168 * np.sin(2 * np.pi * t / 168 + phase_168)
        )

        # Rectify (no negative commits) and add Poisson noise
        signal = np.clip(signal, 0, None)
        signal = rng.poisson(signal).astype(np.float64)
        signals[name] = signal

    logger.info("Generated synthetic signals for %d developers (%d hours).",
                n_developers, hours)
    return signals


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_data(use_bigquery: bool = False, **kwargs) -> Dict[str, np.ndarray]:
    """High-level loader: BigQuery when available, otherwise synthetic demo."""
    if use_bigquery:
        raw = fetch_from_bigquery(**kwargs)
        return build_hourly_timeseries(raw)
    return generate_synthetic_data(**kwargs)


# ---------------------------------------------------------------------------
# Public GitHub API — single-user event fetcher (no auth required)
# ---------------------------------------------------------------------------

_GITHUB_MAX_PAGES = 10  # GitHub API hard limit for the Events endpoint

def fetch_github_events(
    username: str,
    max_pages: int = _GITHUB_MAX_PAGES,
) -> List[dict]:
    """Fetch public events for a GitHub user via the REST API.

    No authentication required.  Rate limit: 60 requests/hour per IP.
    The Events endpoint allows at most 10 pages of 30 events (300 total).
    """
    max_pages = min(max_pages, _GITHUB_MAX_PAGES)
    all_events: List[dict] = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://api.github.com/users/{username}"
            f"/events?page={page}&per_page=30"
        )
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", "FourierPsychology/1.0")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                if not data:
                    break
                all_events.extend(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"GitHub user '{username}' not found.")
            elif e.code == 403:
                raise ValueError(
                    "GitHub API rate limit exceeded (60 req/hour without auth). "
                    "Try again later."
                )
            # 422 or any other HTTP error: stop paginating, keep what we have
            logger.warning(
                "HTTP %d on page %d for %s — stopping pagination.",
                e.code, page, username,
            )
            break

    logger.info("Fetched %d public events for %s.", len(all_events), username)
    return all_events


def build_signal_from_events(
    events: List[dict],
) -> Tuple[np.ndarray, str, str, int]:
    """Convert GitHub API events into an hourly signal array.

    Parameters
    ----------
    events : list[dict]
        Raw events from ``fetch_github_events``.

    Returns
    -------
    signal : np.ndarray
        Zero-filled hourly event-count array.
    start_date : str
        ISO date of first event.
    end_date : str
        ISO date of last event.
    start_weekday : int
        Weekday of the first hour (0 = Monday, 6 = Sunday).
    """
    timestamps = []
    for e in events:
        if "created_at" in e:
            timestamps.append(pd.to_datetime(e["created_at"], utc=True))

    if not timestamps:
        raise ValueError("No events with timestamps found.")

    timestamps.sort()

    hour_min = timestamps[0].floor("h")
    hour_max = timestamps[-1].floor("h")

    full_range = pd.date_range(start=hour_min, end=hour_max, freq="h")

    # Count events per hour
    hour_bins = pd.Series([ts.floor("h") for ts in timestamps])
    counts = hour_bins.value_counts().reindex(full_range, fill_value=0).sort_index()

    signal = counts.values.astype(np.float64)
    start_date = hour_min.strftime("%Y-%m-%d")
    end_date = hour_max.strftime("%Y-%m-%d")
    start_weekday = hour_min.weekday()

    logger.info(
        "Built signal: %d hours (%s to %s), %d total events.",
        len(signal), start_date, end_date, int(signal.sum()),
    )
    return signal, start_date, end_date, start_weekday
