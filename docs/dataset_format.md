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

## Split Data

Create train, validation, and test CSVs with:

```bash
python ipi.py split-data \
  --input-file data/annotated_documents.jsonl \
  --output-dir data/splits \
  --seed 42 \
  --stratify-by constrained_min_labels \
  --min-label-presence 1
```

Use `python ipi.py split-data --help` to see ratio, stratification, chunking, manifest, and output-format options. When token chunking is enabled, also provide `--chunk-max-length` and `--chunk-tokenizer-model`.

## Validate Before Training

Check parsing, offset bounds, overlaps, and tokenizer alignment:

```bash
python ipi.py validate-data --config configs/train.example.yaml
```

Check that every validation and test label is represented in training:

```bash
python ipi.py label-coverage --config configs/train.example.yaml
```

The model schema is derived from training labels. A validation or test label absent from training cannot be learned and should be fixed by changing the split or adding suitable training examples.
