from pathlib import Path
from typing import Tuple

import torch

from resnet_with_groupnorm import ResNet


class KD(torch.nn.Module):
    def __init__(self, num_classes: int, teacher_saved_model: Path, resnet_planes: Tuple[int] = (32, 64, 128, 256),
                 resnet_layers: Tuple[int] = (1, 1, 1, 1), teacher_model_state_dict_key: str = "model_state_dict"):
        """
        Create the knowledge distillation model.

        Args:
            num_classes: Number of output classes for final layer
            teacher_saved_model: Checkpoint for teacher model
            resnet_planes: Number of input planes in ResNet model
            resnet_layers: Number of layers in ResNet model
            teacher_model_state_dict_key: Key in teacher model state dictionary to search for
        """
        super(KD, self).__init__()

        self.teacher_model = ResNet(planes=resnet_planes, layers=resnet_layers, num_classes=num_classes)
        self.student_model = ResNet(planes=resnet_planes, layers=resnet_layers, num_classes=num_classes)

        # Load the teacher model weights
        ckpt = torch.load(f=teacher_saved_model)[teacher_model_state_dict_key]
        self.teacher_model.load_state_dict(state_dict=ckpt, strict=False)

        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            teacher_out = self.teacher_model(x[0])

        student_out = self.student_model(x[1])

        return teacher_out, student_out
