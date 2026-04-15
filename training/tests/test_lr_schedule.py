from __future__ import annotations

import math


def scheduled_lrs_for_epoch(
    get_lr, *, epochs: int, iters: int, accumulation_steps: int, learning_rate: float
) -> list[float]:
    total_update_steps = epochs * math.ceil(iters / accumulation_steps)
    lrs: list[float] = []
    optimizer_step = 0
    for _epoch in range(epochs):
        for step in range(1, iters + 1):
            update_due = step % accumulation_steps == 0 or step == iters
            if not update_due:
                continue
            next_update_step = optimizer_step + 1
            lrs.append(get_lr(next_update_step, total_update_steps, learning_rate))
            optimizer_step += 1
    return lrs


def test_schedule_advances_per_optimizer_update_not_per_micro_batch(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    get_lr = trainer_utils.get_lr
    lrs = scheduled_lrs_for_epoch(get_lr, epochs=1, iters=8, accumulation_steps=4, learning_rate=1.0)

    assert len(lrs) == 2
    assert lrs[0] == get_lr(1, 2, 1.0)
    assert lrs[1] == get_lr(2, 2, 1.0)


def test_schedule_length_scales_with_optimizer_updates(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    get_lr = trainer_utils.get_lr
    lrs = scheduled_lrs_for_epoch(get_lr, epochs=2, iters=10, accumulation_steps=4, learning_rate=0.01)

    expected_updates = 2 * math.ceil(10 / 4)
    assert len(lrs) == expected_updates
    assert lrs[-1] == get_lr(expected_updates, expected_updates, 0.01)


def test_constant_schedule_stays_flat_after_warmup(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    get_lr = trainer_utils.get_lr

    assert get_lr(1, 10, 1.0, schedule="constant", warmup_steps=2) == 0.5
    assert get_lr(2, 10, 1.0, schedule="constant", warmup_steps=2) == 1.0
    assert get_lr(5, 10, 1.0, schedule="constant", warmup_steps=2) == 1.0


def test_cosine_schedule_respects_min_ratio(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    get_lr = trainer_utils.get_lr

    assert get_lr(10, 10, 2.0, schedule="cosine", min_lr_ratio=0.25) == 0.5
