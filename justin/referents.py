"""Features of the channels a question or its options name, for comparisons the encoders blur.

A question such as "Did the metric 'a_cpu' change before the metric 'b_mem'?" or an option set
of service names refers to particular channels. Resolving those names to channels and reading
their deviation features (``deviation.py``) gives the model the sizes and onset times it needs
to compare them, the way a rule would, without it having to find the named channels itself.

An entity is the deviation features of the strongest channel a name resolves to, a flag that
says whether the name resolved at all, and a summary over all the channels it resolves to (how
many deviate, their mean size, the earliest start): a service whose metrics mostly deviate is
more likely the cause than one with a single deviating metric. A name is a channel
(``cartservice_cpu``) or a service whose channels share the prefix before the first underscore
(``cartservice``). The question also carries the same kind of summary over every channel of the
window.
"""

import math
import re
from datetime import UTC, datetime

import numpy as np

from .deviation import N_FEATURES

FOUND = N_FEATURES  # index of the "resolved" flag within an entity
AGG = 6
ENTITY = N_FEATURES + 1 + AGG
CONTEXT = 4
EXTRA = 8  # what an option's own text says about times and numbers
OPTION = ENTITY + EXTRA
PAIR = 5
QUESTION = 2 * ENTITY + PAIR + CONTEXT
_QUOTED = re.compile(r"'([^']+)'")


def resolve(text: str, names: tuple[str, ...]) -> list[int]:
    """Indices of the channels `text` names: an exact channel name, else a service prefix."""
    text = text.strip()
    exact = [i for i, name in enumerate(names) if name == text]
    return exact or [i for i, name in enumerate(names) if name.split("_", 1)[0] == text]


def aggregate(indices: list[int], features: np.ndarray) -> np.ndarray:
    """[AGG]: how many channels there are and deviate, their mean size, and the earliest start."""
    rows = features[indices]
    found = rows[:, 4] > 0
    return np.array(
        [
            min(len(rows), 20) / 20, found.mean(), min(int(found.sum()), 20) / 20,
            rows[:, 1].mean(), rows[:, 0].mean(), rows[found, 3].min() if found.any() else 1.0,
        ],
        np.float32,
    )  # fmt: skip


def entity(indices: list[int], features: np.ndarray) -> np.ndarray:
    """[ENTITY]: the strongest channel's features, a found flag, and `aggregate` of all channels."""
    if not indices:
        return np.zeros(ENTITY, np.float32)
    best = max(indices, key=lambda i: features[i, 1])
    return np.concatenate([features[best], [1.0], aggregate(indices, features)]).astype(np.float32)


def context(features: np.ndarray) -> np.ndarray:
    """[CONTEXT]: over every channel of the window, the share deviating, the largest and mean size,
    and how many channels there are."""
    found = features[:, 4] > 0
    return np.array(
        [found.mean(), features[:, 1].max(), features[:, 1].mean(), min(len(features), 100) / 100],
        np.float32,
    )


_TIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_RANGE = re.compile(r"The time range is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) to (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_LOG_SCALE = 8.0


def _epoch(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def summary(features: np.ndarray, indices: list[int]) -> np.ndarray:
    """[onset fraction, end fraction, tail deviation, max deviation] over the deviating channels."""
    rows = features[[i for i in indices if features[i, 4] > 0]]
    if not len(rows):
        return np.array([1.0, 1.0, 0.0, 0.0], np.float32)
    end = rows[:, 7].max() if (rows[:, 8] > 0).all() else 1.0
    return np.array([rows[:, 3].min(), end, rows[:, 0].max(), rows[:, 1].max()], np.float32)


def option_extras(option: str, question: str, facts: np.ndarray) -> np.ndarray:
    """[EXTRA]: a timestamp option as a position in the window and its gaps to the deviation's
    start and end, a number option as a log size and its gaps to the deviation's size.

    `facts` is `summary(...)` of the series the question is about.
    """
    out = np.zeros(EXTRA, np.float32)
    text = option.strip()
    window = _RANGE.search(question)
    if _TIME.match(text) and window:
        start, end = _epoch(window.group(1)), _epoch(window.group(2))
        fraction = (_epoch(text) - start) / max(end - start, 1.0)
        out[:4] = [1.0, fraction, fraction - facts[0], fraction - facts[1]]
    elif _NUMBER.match(text):
        size = math.log1p(abs(float(text))) / _LOG_SCALE
        out[4:] = [1.0, size, size - facts[2], size - facts[3]]
    return out


def question_vector(a: np.ndarray, b: np.ndarray, window: np.ndarray | None = None) -> np.ndarray:
    """[QUESTION]: two entities, what separates them (see `referent_features`), and the window."""
    both, timed = a[FOUND] * b[FOUND], a[FOUND] * b[FOUND] * a[4] * b[4]
    pair = np.array(
        [both, timed, timed * 5 * (a[3] - b[3]), timed * (a[5] - b[5]), both * (a[1] - b[1])],
        np.float32,
    )
    return np.concatenate([a, b, pair, np.zeros(CONTEXT, np.float32) if window is None else window])


def referent_features(
    question: str, options: tuple[str, ...], names: tuple[str, ...], features: np.ndarray
) -> dict:
    """``referent_question`` [QUESTION] and ``referent_options`` [n_options, ENTITY].

    The question vector holds the entities for up to two quoted names and, for a pair, their
    differences (onset gap as a fraction of the window and in log seconds, and deviation gap),
    signed so a negative gap means the first name changed first.
    """
    features = np.asarray(features, dtype=np.float32)
    quoted = [entity(resolve(q, names), features) for q in _QUOTED.findall(question)[:2]]
    quoted += [np.zeros(ENTITY, np.float32)] * (2 - len(quoted))
    return {
        "referent_question": question_vector(*quoted, context(features)),
        "referent_options": np.asarray(
            [np.concatenate([entity(resolve(o, names), features), np.zeros(EXTRA, np.float32)]) for o in options],
            np.float32,
        ).reshape(len(options), OPTION),
    }


_SERIES_PREFIX = re.compile(r"^time-series (\d+): ")


def grouped_referents(
    options: tuple[str, ...], names: tuple[str, ...], features: np.ndarray, question: str = ""
) -> dict:
    """Referents for questions that refer to "time-series 1" and "time-series 2", as in ARFBench.

    Channels of a two-series question are named ``time-series N: label``. The question's
    entities are the strongest channel of series 1 and of series 2 (of the one series, for a
    single-series question, with the second entity empty). An option names channels by their
    labels, several separated by ", " ("partition:5, partition:11"), and its entity is the
    strongest of them; options that are timestamps or numbers also carry how they relate to
    the series' deviation (`option_extras`).
    """
    features = np.asarray(features, dtype=np.float32)
    series = [int(m.group(1)) if (m := _SERIES_PREFIX.match(n)) else 1 for n in names]
    groups = [[i for i, g in enumerate(series) if g == number] for number in sorted(set(series))]
    first, second = (entity(groups[0], features), entity(groups[1], features)) if len(groups) > 1 \
        else (entity(groups[0], features), np.zeros(ENTITY, np.float32))  # fmt: skip
    facts = summary(features, groups[0])
    labels = [_SERIES_PREFIX.sub("", n) for n in names]
    rows = []
    for option in options:
        hits = [i for i, label in enumerate(labels) if label in {p.strip() for p in option.split(", ")}]
        rows.append(np.concatenate([entity(hits, features), option_extras(option, question, facts)]))
    return {
        "referent_question": question_vector(first, second, context(features)),
        "referent_options": np.asarray(rows, np.float32).reshape(len(options), OPTION),
    }
