# Justin

Time-series question answering with direct Choice, Noul (yes/no) and Score outputs. No generated text.

A frozen time-series foundation model (**Chronos-2**) and a frozen MiniLM text encoder feed a small trainable
cross-attention head. LoRA adapters can unfreeze the top blocks of the series encoder.

## Versions

One GitHub repo (`justin`) holds the code for every version. Each backbone gets its own Hugging Face model repo,
`justin-<backbone>`, and each release is a git tag `vX.Y.Z` with a checkpoint named
`justin-vX.Y.Z-<backbone>-<setup>.safetensors`. A new backbone means a new `SeriesEncoder` (see below) and a new HF repo.

| version | backbone | HF repo | checkpoint | setup |
|---|---|---|---|---|
| `v0.1.0` | Chronos-2 (`amazon/chronos-2`, Apache-2.0) | `justin-chronos` | `justin-v0.1.0-chronos2-minilm-lora4-skip.safetensors` (33 MB) | MiniLM text encoder, `signal_skip`, LoRA (rank 8) on the top 4 of 12 Chronos-2 blocks, trained jointly on HAR, sleep, TSQA and ECG |

A published file holds only what this project trained (fusion, heads, `signal_skip`, LoRA adapters), not the
frozen backbones. It records the exact Hub revisions of those backbones, and loading downloads exactly those:

| frozen backbone | repo | revision |
|---|---|---|
| Chronos-2 | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

### v0.1.0 results

Test split, one checkpoint (the best pooled validation epoch), balanced accuracy / macro-F1 / accuracy in %.
One training run, no seeds; epoch-to-epoch noise on validation is about 1-2 points.

| task | n (test) | bal-acc | macro-F1 | acc | OpenTSLM F1 / acc* |
|---|---|---|---|---|---|
| HAR | 8,222 | 66.8 | 67.1 | 69.9 | 65.4 / 71.5 |
| Sleep | 2,765 | 72.8 | 68.0 | 92.3 | 69.9 / 81.1 |
| ECG-QA | 41,093 | 42.3 | 37.6 | 52.4 | 40.3 / 46.3 |
| TSQA | 4,793 | 99.8 | 99.8 | 99.8 | 97.5 / 97.5 |

\* OpenTSLM's best variant per task, from Langer et al., [*OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data*](https://arxiv.org/abs/2510.02410) (arXiv:2510.02410, 2025). The HAR and sleep F1 values (65.4, 69.9) match the paper's abstract; the other figures are as I recorded them from its tables and have **not been re-checked**.

Read the comparison with care:

- OpenTSLM generates a rationale before answering; Justin picks among supplied options.
- Only **HAR** and **ECG** use OpenTSLM's original train / validation / test files. **Sleep** (split by subject) and
  **TSQA** (split by signal) use splits defined in this repo, so those rows are not like-for-like. The sleep test
  split is very imbalanced (92% accuracy, 73% balanced accuracy).
- Accuracy and macro-F1 disagree on sleep and ECG because of class imbalance. Overall it is roughly on par with
  OpenTSLM, ahead on TSQA, behind on ECG macro-F1.
- The 4-block lead over a 2-block LoRA run is within validation noise; training overfits after epoch 2.

## Use a trained model

```sh
uv sync
```

```python
import torch
from src.pretrained import answer, load_pretrained

# a local file, or a Hugging Face repo id plus the filename
model, config = load_pretrained(
    "<owner>/justin-chronos", filename="justin-v0.1.0-chronos2-minilm-lora4-skip.safetensors"
)

series = torch.randn(3, 128)  # [channels, time]
answer(model, series, "What activity is shown?", ["walking", "sitting", "lying"])  # -> one option
answer(model, series, "Is the signal rising?")                                      # -> True / False
```

`SystemOne.forward(series [B,C,T], questions, options=..., head="choice" | "noul" | "score")` returns raw logits.
Choice can only select one of the supplied option strings; include relevant metadata in the question text.
The first load also downloads the frozen Chronos-2 and MiniLM weights, at the revisions recorded in the file.
`model.temperatures` holds the fitted calibration temperatures per head (see `src.training.calibration.probabilities`).

The file is safetensors with the model config, revisions and temperatures in its metadata (also written to a `.json`
beside it). A full training checkpoint (`.pt`) loads too. To publish a trained checkpoint:

```sh
uv run python -m src export runs/model.pt justin-v0.1.0-chronos2-minilm-lora4-skip.safetensors
```

Export checks that every frozen tensor in the checkpoint equals the pinned backbone and fails otherwise, so a wrong
pin is caught before release. To move to newer backbones, change `revision` on `Chronos2Encoder` / `TextEncoder`
(`src/model/encoders.py`) and retrain; old files keep loading the revisions they recorded.

## How a question is answered

`B` = batch, `C` = channels, `T` = time samples, `M` = signal tokens, `N` = supplied answer options.

```text
series [B,C,T] --> SeriesEncoder --> signal tokens [B,M,d_ts] + mask
                    Chronos-2: d=768, patch 16     (LoRA on the top blocks)
question text --> MiniLM (frozen) --> question tokens [B,L,384]

CrossAttentionFusion: question tokens attend over signal tokens (hidden_dim, `fusion_layers` blocks),
                      masked mean over the question tokens, plus a residual from the mean question vector
  + signal_skip: LayerNorm(masked mean of signal tokens) -> Linear, added to the fused vector
  -> fused vector z [B,hidden_dim]

Choice: options [N strings] -> MiniLM -> projection -> keys [N,hidden]; logits = z @ keys.T / sqrt(hidden)
Noul:   z -> Linear -> one logit (yes if >= 0)
Score:  z -> Linear -> K level logits
```

### Series encoders

| `series_encoder` | model | params | token layout per channel |
|---|---|---|---|
| `chronos` | amazon/chronos-2 (Apache-2.0), encoder re-implemented in `src/model/chronos2.py` | 116M | N patches + `[REG]` + one future-patch token |

An encoder implements the `SeriesEncoder` base class (`src/model/encoders.py`): `forward`, `encode_prefix` /
`encode_suffix` (split at a block boundary so the frozen prefix can be cached), `num_blocks`, `lora_targets`.
Register it in `SERIES_ENCODERS` to make it selectable.

## Datasets

All four come from the OpenTSLM CoT releases and are Choice/Noul tasks over supplied options.

| name | signal | task | split |
|---|---|---|---|
| `har` | 3-channel accelerometer | 8 activities | original train / val / test files |
| `sleep` | 1-channel EEG epoch | 5 sleep stages | own split by subject (seeded 80/10/10) |
| `tsqa` | 1-channel series | multiple-choice trend/pattern questions | own split by signal identity |
| `ecg` | 12-lead PTB-XL | ECG-QA (templated questions, Noul for yes/no) | original train / val / test files |

Datasets are downloaded from their original sources; check each source's terms before reuse.

## Train and evaluate

```sh
uv run python -m src download CONFIG.yaml     # train + validation files for the configured datasets
uv run python -m src cache CONFIG.yaml        # optional: build the feature cache
uv run python -m src train CONFIG.yaml        # train, calibrate, save a checkpoint
uv run python -m src evaluate CKPT... --split test [--dataset har] [--device cuda] [--download] [--output out.json]
uv run pytest -q && uv run ruff check src tests
```

Every YAML key of `CONFIG.yaml` is a field of `Config` in `src/config.py` (datasets, `series_encoder`,
`series_lora_layers`, `signal_skip`, `hidden_dim`, `epochs`, `cache`, `output`, ...). The v0.1.0 setup is
`signal_skip: true`, `series_lora_layers: 4`, `lora_rank: 8`, `lora_learning_rate: 3e-5`, all four datasets, with a
`cache` directory (about 20 GB of RAM). Checkpoints are selected by pooled validation balanced accuracy, and the
test split is for finished models only.

## Repository map

```text
src/model/        encoders.py (SeriesEncoder, Chronos-2, MiniLM), chronos2.py, fusion.py, lora.py, model.py (SystemOne)
src/pretrained.py load_pretrained / answer;  src/export.py export to safetensors
src/cache/        feature cache, cached datasets and collation
src/training/     run.py, train.py, evaluate.py, loss.py, optimizer.py, ...
src/data/         the four dataset adapters
tests/            offline tests with tiny random encoders (no network)
```

## License

The code is licensed under Apache-2.0 (see `LICENSE`).

## Third-party models

Chronos-2 (`amazon/chronos-2`, Apache-2.0) and `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) are downloaded
from Hugging Face at pinned revisions and are not redistributed in this repository or in the published files.

## References

- P. Langer, T. Kaar, M. Rosenblattl, M. A. Xu, W. Chow, M. Maritsch, R. Jakob, N. Wang, J. Liu, A. Verma, B. Han,
  D. S. Kim, H. Chubb, S. Ceresnak, A. Zahedivash, A. T. S. Sandhu, F. Rodriguez, D. McDuff, E. Fleisch, O. Aalami,
  F. Barata, P. Schmiedmayer. *OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and
  Time-Series Data.* arXiv:2510.02410, 2025. <https://arxiv.org/abs/2510.02410>. Source of the HAR-CoT, Sleep-CoT and
  ECG-QA-CoT datasets and of the comparison figures above.
