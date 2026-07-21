from __future__ import annotations

import sys
import types
from contextlib import contextmanager

from ipi_training.config import TrainingConfig
from ipi_training.training import make_trainer, make_training_args


@contextmanager
def patched_transformers_module(module):
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = module
    try:
        yield
    finally:
        if original is None:
            del sys.modules["transformers"]
        else:
            sys.modules["transformers"] = original


def test_make_training_args_accepts_eval_strategy_alias():
    captured_kwargs = {}

    class FakeTrainingArguments:
        def __init__(
            self,
            output_dir,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            num_train_epochs,
            learning_rate,
            weight_decay,
            seed,
            logging_dir,
            logging_steps,
            gradient_accumulation_steps,
            fp16,
            save_total_limit,
            run_name,
            warmup_ratio,
            dataloader_num_workers,
            report_to,
            eval_strategy,
            save_strategy,
            load_best_model_at_end,
            metric_for_best_model,
            greater_is_better,
        ):
            captured_kwargs.update(locals())

    fake_module = types.SimpleNamespace(TrainingArguments=FakeTrainingArguments)
    config = TrainingConfig(train_file="data/train.csv", validation_file="data/validation.csv", test_file="data/test.csv")

    with patched_transformers_module(fake_module):
        make_training_args(config, "outputs/demo", evaluation_enabled=True)

    assert captured_kwargs["eval_strategy"] == "epoch"
    assert "self" in captured_kwargs


def test_make_training_args_raises_for_unsupported_required_arguments():
    class FakeTrainingArguments:
        def __init__(
            self,
            output_dir,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            num_train_epochs,
            learning_rate,
            weight_decay,
            seed,
        ):
            self.output_dir = output_dir

    fake_module = types.SimpleNamespace(TrainingArguments=FakeTrainingArguments)
    config = TrainingConfig(train_file="data/train.csv", validation_file="data/validation.csv", test_file="data/test.csv")

    with patched_transformers_module(fake_module):
        try:
            make_training_args(config, "outputs/demo", evaluation_enabled=True)
        except RuntimeError as exc:
            assert "TrainingArguments does not support the required arguments" in str(exc)
        else:
            raise AssertionError("Expected make_training_args to fail for unsupported arguments.")


def test_make_trainer_prefers_processing_class_when_supported():
    captured_kwargs = {}

    class FakeTrainer:
        def __init__(self, model=None, args=None, processing_class=None, data_collator=None, compute_metrics=None):
            captured_kwargs.update(locals())

    fake_module = types.SimpleNamespace(Trainer=FakeTrainer)

    with patched_transformers_module(fake_module):
        make_trainer(
            model="model",
            args="args",
            tokenizer="tokenizer",
            data_collator="collator",
            compute_metrics="metrics",
        )

    assert captured_kwargs["processing_class"] == "tokenizer"


def test_make_trainer_uses_tokenizer_when_processing_class_is_unavailable():
    captured_kwargs = {}

    class FakeTrainer:
        def __init__(self, model=None, args=None, tokenizer=None, data_collator=None, compute_metrics=None):
            captured_kwargs.update(locals())

    fake_module = types.SimpleNamespace(Trainer=FakeTrainer)

    with patched_transformers_module(fake_module):
        make_trainer(
            model="model",
            args="args",
            tokenizer="tokenizer",
            data_collator="collator",
            compute_metrics="metrics",
        )

    assert captured_kwargs["tokenizer"] == "tokenizer"
