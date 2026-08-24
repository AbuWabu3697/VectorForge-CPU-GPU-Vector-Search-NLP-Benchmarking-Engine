from __future__ import annotations

import torch
from torch import nn


def minimum_ctc_timesteps(target: torch.Tensor) -> int:
    """Return label length plus extra steps needed between adjacent repeats."""
    if target.numel() < 2:
        return int(target.numel())
    repeats = int((target[1:] == target[:-1]).sum().item())
    return int(target.numel()) + repeats


def validate_ctc_lengths(
    targets: torch.Tensor, target_lengths: torch.Tensor, input_lengths: torch.Tensor
) -> None:
    offset = 0
    for batch_index, (target_length, input_length) in enumerate(
        zip(target_lengths.tolist(), input_lengths.tolist())
    ):
        target = targets[offset : offset + target_length]
        required = minimum_ctc_timesteps(target)
        if required > input_length:
            raise ValueError(
                f"Invalid CTC sample at batch index {batch_index}: target requires "
                f"{required} time steps (including adjacent repeats), model emits {input_length}."
            )
        offset += target_length
    if offset != targets.numel():
        raise ValueError("Sum of target_lengths does not match concatenated target count.")


class CTCLossWithValidation(nn.Module):
    """CTC loss for log probabilities shaped ``[T, B, C]``.

    CTC learns image-to-text alignment without per-character pixel boxes. Input
    lengths describe available model time steps; target lengths describe label
    characters. Invalid samples raise instead of being hidden as zero loss.
    """

    def __init__(self, blank_index: int = 0) -> None:
        super().__init__()
        self.loss = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=False)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("CTC logits must have shape [T, B, C].")
        time_steps, batch_size, _ = logits.shape
        if len(target_lengths) != batch_size:
            raise ValueError("target_lengths must contain one value per batch item.")
        if input_lengths is None:
            input_lengths = torch.full(
                (batch_size,), time_steps, dtype=torch.long, device="cpu"
            )
        input_lengths = input_lengths.to(dtype=torch.long, device="cpu")
        cpu_target_lengths = target_lengths.to(dtype=torch.long, device="cpu")
        validate_ctc_lengths(targets.detach().cpu(), cpu_target_lengths, input_lengths)
        return self.loss(
            logits.log_softmax(dim=2), targets, input_lengths, cpu_target_lengths
        )
