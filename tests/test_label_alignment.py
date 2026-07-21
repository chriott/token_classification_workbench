from ipi_training.labels import LabelSchema, build_label_aligner


class FakeTokenized(dict):
    def char_to_token(self, char_index: int):
        mapping = {
            0: 1,
            1: 1,
            2: 1,
            4: 2,
            5: 2,
            6: 2,
        }
        return mapping.get(char_index)


class FakeTokenizer:
    def __call__(self, text, truncation, max_length, return_offsets_mapping, return_attention_mask):
        assert text == "ABC DEF"
        assert truncation is True
        assert max_length == 16
        assert return_offsets_mapping is True
        assert return_attention_mask is True
        return FakeTokenized({"offset_mapping": [(0, 0), (0, 3), (4, 7), (0, 0)]})


def test_build_label_aligner_assigns_bio_tags():
    schema = LabelSchema(
        labels=["IPI-NAME"],
        bio_labels=["O", "B-IPI-NAME", "I-IPI-NAME"],
        label_to_id={"O": 0, "B-IPI-NAME": 1, "I-IPI-NAME": 2},
        id_to_label={0: "O", 1: "B-IPI-NAME", 2: "I-IPI-NAME"},
    )
    align = build_label_aligner(FakeTokenizer(), schema, max_length=16)

    tokenized = align(
        {
            "text": "ABC DEF",
            "spans": [{"start": 0, "end": 3, "label": "IPI-NAME"}],
        }
    )

    assert tokenized["labels"] == [-100, 1, 0, -100]
