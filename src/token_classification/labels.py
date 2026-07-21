from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class LabelSchema:
    labels: List[str]
    bio_labels: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]

    @classmethod
    def from_dataset(cls, dataset) -> "LabelSchema":
        labels = set()
        for example in dataset:
            for span in example.get("spans", []):
                label = span.get("label")
                if label:
                    labels.add(label)
        sorted_labels = sorted(labels)
        bio_labels = ["O"] + [f"{prefix}-{label}" for label in sorted_labels for prefix in ("B", "I")]
        label_to_id = {label: idx for idx, label in enumerate(bio_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        return cls(sorted_labels, bio_labels, label_to_id, id_to_label)


def build_label_aligner(tokenizer, schema: LabelSchema, max_length: int):
    def align_labels_with_tokens(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        labels = []
        for start, end in tokenized["offset_mapping"]:
            labels.append(-100 if start == 0 and end == 0 else schema.label_to_id["O"])

        unmapped = []
        for span in example.get("spans", []):
            start_char = span.get("start")
            end_char = span.get("end")
            label = span.get("label")
            if label not in schema.labels or start_char is None or end_char is None:
                continue
            token_start = tokenized.char_to_token(start_char)
            token_end = tokenized.char_to_token(end_char - 1)
            if token_start is None or token_end is None:
                unmapped.append(span)
                continue
            if labels[token_start] != -100:
                labels[token_start] = schema.label_to_id[f"B-{label}"]
            for index in range(token_start + 1, token_end + 1):
                if labels[index] != -100:
                    labels[index] = schema.label_to_id[f"I-{label}"]
        if unmapped:
            print(
                f"Warning: could not align {len(unmapped)} spans in example "
                f"{example.get('id', 'unknown')}: {unmapped}"
            )
        tokenized["labels"] = labels
        return tokenized

    return align_labels_with_tokens


def bio_to_spans(labels: Sequence[str], offsets: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
    spans = []
    current_label = None
    current_start = None
    current_end = None
    for label, offset in zip(labels, offsets):
        start, end = offset
        if start == 0 and end == 0:
            continue
        if label in (None, "O"):
            if current_label is not None:
                spans.append({"start": current_start, "end": current_end, "label": current_label})
                current_label = None
            continue
        if "-" in label:
            prefix, entity_type = label.split("-", 1)
        else:
            prefix, entity_type = "B", label
        if prefix == "B" or entity_type != current_label:
            if current_label is not None:
                spans.append({"start": current_start, "end": current_end, "label": current_label})
            current_label = entity_type
            current_start = start
            current_end = end
        else:
            current_end = end
    if current_label is not None:
        spans.append({"start": current_start, "end": current_end, "label": current_label})
    return spans


def bio_to_entities(token_spans: Sequence[Sequence[int]], labels: Sequence[str]) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    current_label = ""
    start_char = -1
    end_char = -1

    for index, label in enumerate(labels):
        if index >= len(token_spans):
            break
        if label == "O":
            if current_label:
                entities.append({"label": current_label, "start": start_char, "end": end_char})
                current_label = ""
                start_char = -1
                end_char = -1
            continue
        if "-" in label:
            prefix, entity_type = label.split("-", 1)
        else:
            prefix, entity_type = "B", label
        span_start, span_end = token_spans[index]
        if prefix == "B" or not current_label or entity_type != current_label:
            if current_label:
                entities.append({"label": current_label, "start": start_char, "end": end_char})
            current_label = entity_type
            start_char = span_start
            end_char = span_end
        else:
            current_label = entity_type
            end_char = span_end

    if current_label:
        entities.append({"label": current_label, "start": start_char, "end": end_char})

    return entities
