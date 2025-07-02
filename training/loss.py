import torch


def kd_loss(teacher_output: torch.Tensor, student_output: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute the KL-Divergence loss between teacher and student output features

    Args:
        teacher_output: Teacher model output features
        student_output: Student model output features
        temperature: Temperature value for controlling softness of loss

    Returns:
        KL-Divergence loss for teacher and student output features
    """
    prob_student = torch.nn.LogSoftmax(dim=1)(student_output / temperature)
    prob_teacher = torch.nn.Softmax(dim=1)(teacher_output / temperature)
    return torch.nn.KLDivLoss(reduction="batchmean")(input=prob_student, target=prob_teacher) * (temperature ** 2)


def pixel_loss(teacher_output: torch.Tensor, student_output: torch.Tensor, loss_func: torch.nn.Module,
               resize_mode: str) -> float:
    """
    Compute the pixel-wise loss with specified resizing options to match output sizes of teacher and student models.

    Args:
        teacher_output: Teacher model feature map output
        student_output: Student model feature map output
        loss_func: Loss function for computation (e.g., L1- or L2-loss)
        resize_mode: Resizing mode to use for ensuring same size between teacher and student

    Returns:
        Loss between teacher and student models, pixel-wise.
    """
    loss = 0.0

    output_size = tuple(student_output.shape[2:4])

    if (resize_mode == "maxpool") or (resize_mode == "maxpool+bicubic"):
        loss += loss_func(input=student_output,
                          target=torch.nn.functional.adaptive_max_pool2d(input=teacher_output,
                                                                         output_size=output_size))
    elif (resize_mode == "bicubic") or (resize_mode == "maxpool+bicubic"):
        loss += loss_func(input=student_output,
                          target=torch.nn.functional.interpolate(input=teacher_output, size=output_size,
                                                                 align_corners=False, mode="bicubic"))
    if resize_mode == "maxpool+bicubic":
        loss /= 2

    return loss
