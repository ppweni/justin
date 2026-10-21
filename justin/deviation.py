"""Per-channel deviation features that survive the encoders' own normalization.

Series encoders normalize every channel internally (Toto's causal scaler and asinh, Chronos-2's
instance norm), so a shift of 10 baseline deviations and one of 1000 look alike afterwards.
Questions that compare channels ("which metric deviated most?") need that size, and so do
timing questions in seconds. These features read it off the series before the encoder and are
added to the channel's tokens (``channel_features`` in the config).

The input is a series in baseline units: ``z = (x - baseline mean) / baseline std``.
"""

import numpy as np

N_FEATURES = 9
_LOG_SCALE = 8.0  # log1p(3000): keeps the features near [0, 1]


def baseline_zscore(values: np.ndarray, fraction: float = 0.3) -> np.ndarray:
    """Standardize each channel against its first `fraction` of points (the assumed baseline)."""
    count = max(2, int(values.shape[1] * fraction))
    baseline = values[:, :count]
    spread = np.maximum(baseline.std(axis=1), 0.05 * values.std(axis=1)) + 1e-9
    return (values - baseline.mean(axis=1, keepdims=True)) / spread[:, None]


def normalize(values: np.ndarray, baseline: float | tuple[int, int] | None = 0.3) -> np.ndarray:
    """Baseline units for a raw ``[channels, time]`` array.

    ``baseline`` is the share of the series taken as normal behavior (the first 30% by default),
    a ``(start, end)`` index range of normal behavior, or ``None`` when `values` are already in
    baseline units.
    """
    values = np.asarray(values, dtype=np.float64)
    if baseline is None:
        return values
    if isinstance(baseline, tuple):
        start, end = baseline
        if not 0 <= start < end - 1 <= values.shape[1]:
            raise ValueError("baseline must be a range of at least two points inside the series")
        reference = values[:, start:end]
        spread = np.maximum(reference.std(axis=1), 0.05 * values.std(axis=1)) + 1e-9
        return (values - reference.mean(axis=1, keepdims=True)) / spread[:, None]
    if not 0 < baseline < 1:
        raise ValueError("a baseline share must be between 0 and 1")
    return baseline_zscore(values, baseline)


def deviation_features(z: np.ndarray, step: float) -> np.ndarray:
    """[C, 9] features of `z` [C, T], with `step` seconds between points.

    log size of the deviation in the last 40% of the series (median) and at its maximum, the same
    for the first 40%, the position of the first sustained jump (three points beyond 3 baseline
    deviations) as a fraction and in seconds, the length of the series in seconds, and where the
    last sustained deviation ends (a fraction; 1 when it never does) with a flag that it ends
    at least a few points before the series does.
    """
    z = np.abs(np.asarray(z, dtype=np.float64))
    channels, length = z.shape
    head, tail = z[:, : max(1, int(length * 0.4))], z[:, int(length * 0.6) :]
    beyond = z > 3
    if length >= 3:
        sustained = beyond[:, :-2] & beyond[:, 1:-1] & beyond[:, 2:]
        found = sustained.any(axis=1)
        first = np.where(found, sustained.argmax(axis=1), 0)
        last = np.where(found, sustained.shape[1] - 1 - sustained[:, ::-1].argmax(axis=1), 0) + 3
    else:
        found, first = np.zeros(channels, bool), np.zeros(channels, int)
        last = np.zeros(channels, int)
    ended = found & (length - last >= max(3, 0.03 * length))
    log = lambda x: np.log1p(x) / _LOG_SCALE  # noqa: E731
    return np.stack(
        [
            log(np.median(tail, axis=1)),
            log(z.max(axis=1)),
            log(np.median(head, axis=1)),
            np.where(found, first / length, 1.0),
            found.astype(np.float64),
            log(np.where(found, first * step, 0.0)),
            np.full(channels, log(length * step)),
            np.where(ended, last / length, 1.0),
            ended.astype(np.float64),
        ],
        axis=1,
    ).astype(np.float32)
