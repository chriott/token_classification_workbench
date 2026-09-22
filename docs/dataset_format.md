# Dataset Format

The pipeline accepts CSV, JSON, and JSONL input. Training uses separate train, validation, and test files; `split-data` can create them from one annotated source file.

## Required Fields

Each record requires:

- A text field. `text` is preferred; supported fallbacks include `note_text`, `document`, `doc`, `notes`, and `content`. Set `text_column` in the YAML config to select another column explicitly.
- A span field. `spans` is preferred and `annotations` is supported as a fallback. Set `spans_column` to select it explicitly.

Each span contains a label and character offsets into the exact record text:

```json
{"start": 15, "end": 23, "label": "PERSON"}
```

`start` is inclusive and `end` is exclusive, so the annotated substring is `text[start:end]`. Label names are arbitrary and are learned from the training split.

For CSV input, the span list is stored as a JSON-like or Python-literal string:

```csv
text,spans
"Example text for training","[{""start"": 0, ""end"": 7, ""label"": ""EXAMPLE""}]"
```

For JSONL input, use one object per line:

```json
{"text":"Example text for training","spans":[{"start":0,"end":7,"label":"EXAMPLE"}]}
```

Span objects using `begin` instead of `start` are normalized automatically. If `end` is absent, the loader can derive it from `begin` and `text_span`.

## Optional Metadata

Columns listed under `optional_string_columns` in the training config are preserved in prediction reports when present. For example:

```yaml
optional_string_columns:
  - document_id
  - source
```

Set this to `[]` when no metadata needs to be preserved.

## Excluding Labels

To omit documented labels from a training run without rewriting the source data, list them at the top level of the
training config:

```yaml
excluded_labels:
  - RARE_LABEL
```

Their spans are removed in memory from training, validation, and test data. Rows that contain no remaining spans are
kept as negative examples. Excluded labels therefore do not enter the model label schema or evaluation metrics.

## Split Data

Choose `model_name` and `max_length` in the training config before chunking long documents. Then create train, validation, and test files using the same tokenizer and final sequence length:

```bash
token-classification split-data \
  --input-file data/annotated_documents.jsonl \
  --output-dir data/splits \
  --seed 137 \
  --stratify-by constrained_min_labels \
  --min-label-presence 1 \
  --chunk-max-length 512 \
  --chunk-tokenizer-model FacebookAI/xlm-roberta-base
```

`--chunk-max-length` is the final model input length. The chunker reserves the tokenizer's special tokens and uses the remaining capacity for content, preventing training-time truncation at chunk boundaries. `--chunk-stride` controls content-token overlap and must be smaller than that remaining capacity.

The `--chunk-tokenizer-model` value should match `model_name` in the training config. Omit all chunk options only when every input record already fits within the selected model's maximum length.

Use `token-classification split-data --help` to see ratio, stratification, manifest, stride, and output-format options.

## Validate Before Training

Check parsing, offset bounds, overlaps, and tokenizer alignment:

```bash
token-classification validate-data --config configs/train.example.yaml
```

Check that every validation and test label is represented in training:

```bash
token-classification label-coverage --config configs/train.example.yaml
```

The model schema is derived from training labels. A validation or test label absent from training cannot be learned and should be fixed by changing the split or adding suitable training examples.
