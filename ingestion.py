"""
Spectral Sabermetrics — Part 1: Data Ingestion Pipeline
========================================================
Extracts developer PushEvent activity from the GitHub Archive BigQuery dataset,
aggregates it into zero-filled hourly time-series suitable for FFT analysis.

Includes a demo/fallback mode that synthesises realistic data locally so the
dashboard can be demonstrated without BigQuery credentials.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict

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

_MACRO_BQ_QUERY_TEMPLATE = """
SELECT
    EXTRACT(DATE FROM created_at) AS ds,
    COUNT(*) AS pushes
FROM
    `githubarchive.day.20*`
WHERE
    _TABLE_SUFFIX BETWEEN
        FORMAT_DATE('%y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL {years} YEAR))
        AND FORMAT_DATE('%y%m%d', CURRENT_DATE())
    AND type = 'PushEvent'
    AND repo.name LIKE '{org}/%'
GROUP BY
    ds
ORDER BY
    ds
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
# Macro-Sabermetrics Data Loader
# ---------------------------------------------------------------------------

def fetch_macro_data(
    org_pattern: str = "microsoft",
    ticker: str = "MSFT",
    years: int = 3,
    use_bigquery: bool = False,
) -> pd.DataFrame:
    """Fetch org-wide GitHub push counts and merge with historical stock prices.

    Parameters
    ----------
    org_pattern : str
        Organization prefix to match repositories (e.g., 'microsoft').
    ticker : str
        Stock ticker symbol for yfinance.
    years : int
        Number of historical years to fetch.
    use_bigquery : bool
        If True, executes BigQuery SQL. If False, generates synthetic fallback.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date sequence with ['pushes', 'Close'].
    """
    import yfinance as yf

    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=years * 365)
    
    logger.info("Downloading %s stock data from %s to %s", ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Handle yfinance multi-index vs single-index depending on version
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_close = stock_df['Close'][ticker].to_frame('Close')
    else:
        stock_close = stock_df[['Close']].copy()

    # Normalise index
    stock_close.index = pd.to_datetime(stock_close.index).tz_localize(None)

    if use_bigquery:
        from google.cloud import bigquery
        client = bigquery.Client()
        query = _MACRO_BQ_QUERY_TEMPLATE.format(org=org_pattern, years=years)
        logger.info("Submitting Macro BigQuery job for org %s over %d years...", org_pattern, years)
        bq_df = client.query(query).to_dataframe()
        bq_df['ds'] = pd.to_datetime(bq_df['ds']).dt.tz_localize(None)
        bq_df.set_index('ds', inplace=True)
    else:
        # Synthetic macroscopic org data
        logger.info("Generating synthetic Macro BigQuery data for %s", org_pattern)
        idx = pd.date_range(start=start_date, end=end_date, freq='d')
        rng = np.random.default_rng(42)
        base = 5000 + 1000 * np.sin(np.linspace(0, 10, len(idx)))  # Trend wave
        # Add weekend dips (lower activity)
        weekend_mask = idx.dayofweek >= 5
        base[weekend_mask] *= 0.3
        pushes = rng.poisson(base).astype(float)
        
        # Add high-frequency noise spikes (stress periods)
        spike_prob = 0.05
        spikes = rng.random(size=len(idx)) < spike_prob
        pushes[spikes] += rng.uniform(2000, 5000, size=spikes.sum())
        
        bq_df = pd.DataFrame({'pushes': pushes}, index=idx)

    # Merge GitHub and stock data (Left join keeps all contiguous calendar days)
    df = bq_df.join(stock_close, how='left')
    
    # Forward-fill stock prices on weekends and holidays
    df['Close'] = df['Close'].ffill()
    
    # Backfill just in case the first couple of days were weekends
    df['Close'] = df['Close'].bfill()
    
    logger.info("Macro data merged. Resulting shape: %s", df.shape)
    
    # Flatten multi-index if it exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df
