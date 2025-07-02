import torch
from tqdm import tqdm
from typing import Callable
from argparse import Namespace


def run_epoch(dl: torch.utils.data.DataLoader, model: torch.nn.Module, is_train: bool, device: torch.device,
              soft_loss: Callable, pixel_loss: Callable, loss_func: torch.nn.Module, temperature: float,
              resize_mode: str, batch_size: int, optimizer: torch.nn.optim.Optimizer = None) -> float:
    """
    Run a single epoch of training or validation examples. Note this uses gradient accumulation.

    Args:
        temperature: Temperature value for soft loss computation
        resize_mode: Resizing functions to use for pixel-wise loss
        batch_size: Number of elements per mini-batch
        dl: Dataloader of mini-batches
        model: Knowledge distillation model
        is_train: Training (True) or validation (False) mode
        device: Device to run on (must be CUDA)
        soft_loss: Function for computing soft portion of loss
        pixel_loss: Function for computing pixel-wise portion of loss
        loss_func: Loss function for pixel-wise loss (e.g., L1- or L2-loss)
        optimizer: Optimizer to use (must be provided for training mode, doesn't matter for validation)

    Returns:
        Loss over one epoch of examples
    """
    model.student_model.train(mode=is_train)

    running_loss = 0.0

    # Reset the gradient
    if is_train:
        assert optimizer is not None, "Optimizer must exist for training mode"
        optimizer.zero_grad()

    for idx, (student_batch, teacher_batch) in enumerate(tqdm(dl, desc="Training" if is_train else "Validation")):
        student_batch = student_batch.to(device)
        teacher_batch = teacher_batch.to(device)

        teacher_out, student_out = model(teacher_batch, student_batch)

        loss = soft_loss(teacher_output=teacher_out[1], student_output=student_out[1],
                         temperature=temperature) + pixel_loss(teacher_output=teacher_out[0],
                                                               student_output=student_out[0], loss_func=loss_func,
                                                               resize_mode=resize_mode)

        running_loss += loss.item()

        loss /= batch_size

        # Gradient accumulation
        if is_train:
            loss.backward()

            # Step optimizer and reset gradient
            if (idx + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

    return running_loss / len(dl)
