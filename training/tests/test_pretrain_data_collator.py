"""Regression checks for the explicit pretrain data collator."""

from __future__ import annotations

import pytest
import torch

datasets = pytest.importorskip("datasets", reason="datasets is required for lm_dataset import")
transformers = pytest.importorskip("transformers", reason="transformers is required for model_minimind import")


def test_pretrain_data_collator_stacks_inputs_and_builds_position_ids(lm_dataset_module, packed_eos_features) -> None:
    collator = lm_dataset_module.PretrainDataCollator(eos_token_id=3, max_seq_len=6)

    batch = collator(packed_eos_features)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    position_ids = batch["position_ids"]
    attention_mask = batch["attention_mask"]
    assert input_ids.shape == (2, 6)
    assert labels.shape == (2, 6)
    assert position_ids.shape == (2, 6)
    assert attention_mask.shape == (2, 6, 6)
    assert torch.equal(input_ids[0], torch.tensor([1, 2, 3, 4, 3, 0]))
    assert torch.equal(input_ids[1], torch.tensor([5, 6, 3, 0, 0, 0]))
    assert torch.equal(labels[0], torch.tensor([1, 2, 3, 4, 3, -100]))
    assert torch.equal(labels[1], torch.tensor([5, 6, 3, -100, -100, -100]))
    assert torch.equal(position_ids[0], torch.tensor([0, 1, 2, 0, 1, 0]))
    assert torch.equal(position_ids[1], torch.tensor([0, 1, 2, 0, 0, 0]))
    assert torch.equal(
        attention_mask[0],
        torch.tensor(
            [
                [True, False, False, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
                [False, False, False, True, False, False],
                [False, False, False, True, True, False],
                [False, False, False, False, False, True],
            ]
        ),
    )


def test_minimind_requires_explicit_position_ids(model_minimind_module) -> None:
    model = model_minimind_module.MiniMindForCausalLM(
        model_minimind_module.MiniMindConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
        )
    )
    input_ids = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="position_ids"):
        model(input_ids)


def test_minimind_accepts_packed_attention_mask(
    lm_dataset_module,
    model_minimind_module,
    packed_eos_features,
) -> None:
    collator = lm_dataset_module.PretrainDataCollator(eos_token_id=3, max_seq_len=6)
    model = model_minimind_module.MiniMindForCausalLM(
        model_minimind_module.MiniMindConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
        )
    )
    batch = collator(packed_eos_features)

    output = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
    )

    assert output.logits.shape[:2] == batch["input_ids"].shape
    assert output.loss is not None


def test_pretrain_data_collator_truncates_oversized_samples_and_logs(lm_dataset_module, capsys) -> None:
    collator = lm_dataset_module.PretrainDataCollator(eos_token_id=99, max_seq_len=4)

    batch = collator([torch.tensor([1, 2, 3, 4, 5])])

    captured = capsys.readouterr()
    assert "truncating sample from length 5 to 4" in captured.out
    assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 3, 99]))
    assert torch.equal(batch["labels"][0], torch.tensor([1, 2, 3, 99]))
    assert torch.equal(batch["position_ids"][0], torch.tensor([0, 1, 2, 3]))
