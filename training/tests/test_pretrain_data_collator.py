"""Regression checks for the explicit pretrain data collator."""

from __future__ import annotations

import pytest
import torch

datasets = pytest.importorskip("datasets", reason="datasets is required for lm_dataset import")
transformers = pytest.importorskip("transformers", reason="transformers is required for model_minimind import")


def _tiny_model_config(model_minimind_module):
    return model_minimind_module.MiniMindConfig(
        hidden_size=32,
        num_hidden_layers=1,
        dropout=0.0,
        vocab_size=128,
        flash_attn=False,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        intermediate_size=64,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        inference_rope_scaling=False,
        num_experts=4,
        num_experts_per_tok=1,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0005,
        head_dim=8,
    )


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
    assert torch.equal(labels[0], torch.tensor([1, 2, 3, -100, 3, -100]))
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
    model = model_minimind_module.MiniMindForCausalLM(_tiny_model_config(model_minimind_module))
    input_ids = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="position_ids"):
        model(input_ids)


def test_minimind_accepts_packed_attention_mask(
    lm_dataset_module,
    model_minimind_module,
    packed_eos_features,
) -> None:
    collator = lm_dataset_module.PretrainDataCollator(eos_token_id=3, max_seq_len=6)
    model = model_minimind_module.MiniMindForCausalLM(_tiny_model_config(model_minimind_module))
    batch = collator(packed_eos_features)

    output = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
    )

    assert output.logits.shape[:2] == batch["input_ids"].shape
    assert output.loss is not None


def test_pretrain_data_collator_omits_mask_for_single_unpacked_rows(lm_dataset_module) -> None:
    collator = lm_dataset_module.PretrainDataCollator(eos_token_id=9, pad_token_id=0, max_seq_len=8)

    batch = collator([torch.tensor([1, 2, 3])])

    assert batch["attention_mask"] is None
    assert torch.equal(batch["position_ids"][0], torch.tensor([0, 1, 2, 3, 0, 0, 0, 0]))


def test_minimind_chunked_loss_matches_full_cross_entropy(model_minimind_module) -> None:
    model = model_minimind_module.MiniMindForCausalLM(_tiny_model_config(model_minimind_module))
    hidden_states = torch.randn(2, 5, model.config.hidden_size)
    labels = torch.tensor([[1, 2, 3, 4, 5], [6, -100, 8, 9, 10]])

    chunked = model_minimind_module.chunked_causal_lm_loss(
        hidden_states,
        labels,
        model.lm_head,
        chunk_size=3,
    )
    logits = model.lm_head(hidden_states)
    full = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )

    assert torch.allclose(chunked, full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton fused CE")
def test_minimind_triton_linear_ce_matches_full_cross_entropy(model_minimind_module) -> None:
    if model_minimind_module.triton is None:
        pytest.skip("triton is not installed")
    torch.manual_seed(0)
    hidden_dim = 32
    vocab_size = 257
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    hidden = torch.randn(2, 7, hidden_dim, device="cuda", requires_grad=True)
    labels = torch.randint(0, vocab_size, (2, 7), device="cuda")
    labels[0, 2] = -100

    loss = model_minimind_module.chunked_causal_lm_loss(hidden, labels, lm_head, chunk_size=3)
    loss.backward()
    hidden_grad = hidden.grad.detach().clone()
    weight_grad = lm_head.weight.grad.detach().clone()

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    ref_head.weight.data.copy_(lm_head.weight.detach())
    ref_logits = ref_head(ref_hidden)
    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits[:, :-1, :].contiguous().view(-1, vocab_size),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    ref_loss.backward()

    assert torch.allclose(loss, ref_loss, atol=1e-5, rtol=1e-5)
    assert torch.allclose(hidden_grad, ref_hidden.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(weight_grad, ref_head.weight.grad, atol=1e-5, rtol=1e-5)


def test_minimind_can_return_loss_without_full_sequence_logits(model_minimind_module) -> None:
    model = model_minimind_module.MiniMindForCausalLM(_tiny_model_config(model_minimind_module))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)

    output = model(
        input_ids,
        position_ids=position_ids,
        labels=input_ids,
        return_full_logits=False,
        loss_chunk_size=2,
    )

    assert output.loss is not None
    assert output.logits.shape[:2] == (1, 1)


def test_minimind_flash_attention_keeps_causal_mask_without_packed_mask(
    model_minimind_module,
    monkeypatch,
) -> None:
    calls = []

    def fake_sdpa(q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False):
        del k, v, dropout_p
        calls.append({"attn_mask": attn_mask, "is_causal": is_causal})
        return torch.zeros_like(q)

    monkeypatch.setattr(model_minimind_module.F, "scaled_dot_product_attention", fake_sdpa)
    config = _tiny_model_config(model_minimind_module)
    config.flash_attn = True
    model = model_minimind_module.MiniMindForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)

    output = model(input_ids, position_ids=position_ids)

    assert output.logits.shape[:2] == input_ids.shape
    assert calls
    assert calls[0] == {"attn_mask": None, "is_causal": True}


def test_pretrain_data_collator_truncates_oversized_samples_and_logs(lm_dataset_module, capsys) -> None:
    collator = lm_dataset_module.PretrainDataCollator(eos_token_id=99, max_seq_len=4)

    batch = collator([torch.tensor([1, 2, 3, 4, 5])])

    captured = capsys.readouterr()
    assert "truncating sample from length 5 to 4" in captured.out
    assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 3, 99]))
    assert torch.equal(batch["labels"][0], torch.tensor([1, 2, 3, 99]))
    assert torch.equal(batch["position_ids"][0], torch.tensor([0, 1, 2, 3]))


def test_pretrain_data_collator_ignores_cross_document_next_token_targets(lm_dataset_module) -> None:
    collator = lm_dataset_module.PretrainDataCollator(eos_token_id=9, pad_token_id=0, max_seq_len=8)

    batch = collator([torch.tensor([1, 2]), torch.tensor([3, 4])])

    assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 9, 3, 4, 9, 0, 0]))
    assert torch.equal(batch["position_ids"][0], torch.tensor([0, 1, 2, 0, 1, 2, 0, 0]))
    assert batch["labels"][0, 3].item() == -100
    assert batch["attention_mask"][0, 3, 2].item() is False
    assert batch["attention_mask"][0, 4, 3].item() is True
