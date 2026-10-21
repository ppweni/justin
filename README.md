# justin

Fast, direct answers to questions about time series: inference for the published Justin models.

Justin is a "System 1" for time series. You give it a multichannel signal, a question and the answers you
allow. It returns one of them (**Choice**), a yes/no (**Noul**) or a level (**Score**), with calibrated
probabilities. It cannot answer outside the options you supply. A **location head** points at where something
happens: a channel and a 32-step patch, or "not present".

- **Series encoder:** Timestral (any of the four sizes, frozen; LoRA adapters and question conditioning
  (FiLM) as trained) or Chronos-2 (frozen, LoRA as trained).
- **Text encoder:** MiniLM, for the question, the options and the channel names.
- **Fusion and heads:** cross-attention fusion with the Choice, Noul and Score heads.

This package is inference only. Training lives in the training repository. A Rust port with the same models is
[`justin_rs`](https://github.com/ppweni/justin_rs).

## Install

```sh
pip install "justin @ git+https://github.com/ppweni/justin@v0.2.0"
```

## Use

```python
import numpy as np
from justin import Justin, Location, Question

repo = "ppweni/justin-timestral-minilm"
name = "justin-v0.2.0-timestral5m-minilm-film8-lora8"
model = Justin.from_pretrained(
    repo, filename=f"{name}.safetensors", revision="v0.2.0",
    location=repo, location_filename=f"{name}-location.safetensors", location_revision="v0.2.0",
)

series = np.random.randn(3, 240).cumsum(-1)  # [channels, time], raw values
names = ["cartservice_cpu", "cartservice_mem", "frontend_latency-50"]

model.answer(series, "Which service is the root cause?", ["cartservice", "frontend", "No service is faulty"], names, step=4.0)
model.probabilities(series, "Is there an anomaly in this time window?", None, names, step=4.0)  # P(yes)
model.locate(series, "Locate the onset of the spike in cartservice_cpu in this window.", names)

# Several queries about one series (the Rust crate's `forward`):
outputs = model.ask(series, [
    Question("Is there an anomaly in this time window?"),           # yes/no
    Question("Which service is the root cause?", ["cartservice", "frontend", "No service is faulty"]),
    Question("How severe is it?", "score"),
    Location("Locate the onset of the spike in cartservice_cpu in this window."),
], names, step=4.0)
```

Arguments:

- **`series`:** `[channels, time]`, or `[time]` for one channel, of raw values.
- **Channel names:** observability models need one name per channel, `<service>_<metric>`.
- **`step`:** the seconds between points.
- **`baseline`:** what counts as normal for the deviation features. The default is the first 30% of the
  window; pass a `(start, end)` index range, or `None` if the series is already in baseline units.

What comes back:

- **Probabilities:** calibrated with the temperatures stored in the file.
- **Locations:** the channel, the patch, and the sample range `start`/`end` the patch covers.

Models load from a local file or a Hugging Face repo (`filename=`). Timestral and MiniLM are downloaded from
their own repositories at the revisions the file records.

## Models

| repo | file | series encoder | trained on |
|---|---|---|---|
| [`ppweni/justin-timestral-minilm`](https://huggingface.co/ppweni/justin-timestral-minilm) | `justin-v0.2.0-timestral5m-minilm-film8-lora8.safetensors` (+ `-location`) | Timestral-5M | microservice incidents (RCAEval, Nezha, AIOps 2020) and time-series QA |
| [`ppweni/justin-chronos-minilm`](https://huggingface.co/ppweni/justin-chronos-minilm) | `justin-v0.1.1-chronos2-minilm-lora4-skip.safetensors` | Chronos-2 | sleep, TSQA, ECG-QA |

The location head needs a Timestral model: Chronos-2's [REG] and future-patch tokens are not positions in the
series.

## Tests

```sh
uv run pytest -q
```

The tests use tiny random models and need no network. `tests/test_parity.py` compares this package with
the training package on the parity fixtures, which the training repository writes. Point `JUSTIN_FIXTURES` at
them (default: `../justin_rs/tests/fixtures`); without fixtures it skips.

## License

The code is Apache-2.0 (see `LICENSE`). `justin/timestral.py` is derived from Mistral AI's Timestral
code (Apache-2.0); the Timestral weights are under Mistral's Modified MIT licence. `justin/chronos2.py`
re-implements the Chronos-2 encoder (Amazon, Apache-2.0). MiniLM is Apache-2.0. No backbone weights are
redistributed here: all are downloaded from their own repositories. Each published Justin model has its own
license on its model card.
