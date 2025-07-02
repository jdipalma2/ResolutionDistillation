import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Resolution Distillation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Training and validation mini-batch size.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Gradient descent learning rate.")
parser.add_argument("--loss_type", type=str, default="l2", choice=("l1", "l2"),
                    help="Specify L1 or L2 loss for optimization.")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for distillation loss.")
parser.add_argument("--gamma", type=float, default=1.0, help="Learning rate decay per epoch.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty) for optimizer.")
parser.add_argument("--num_workers", type=int, default=10, help="Number of subprocesses for IO.")
parser.add_argument("--teacher_saved_model", type=Path, help="Teacher model weights.")
parser.add_argument("--data_dir", type=Path, help="Location of training and validation datasets.")
parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"),
                    help="Path to store model checkpoint files.")
parser.add_argument("--log_dir", type=Path, default=Path("logs"),
                    help="Path to store training and validation metrics to CSV.")
parser.add_argument("--color_jitter_brightness", type=float, default=0.5,
                    help="Brightness range for color jitter transform.")
parser.add_argument("--color_jitter_contrast", type=float, default=0.5,
                    help="Contrast range for color jitter transform.")
parser.add_argument("--color_jitter_saturation", type=float, default=0.5,
                    help="Saturation range for color jitter transform.")
parser.add_argument("--color_jitter_hue", type=float, default=0.2, help="Hue range for color jitter transform.")
parser.add_argument("--ds_ext", type=Path, default=Path("5x"), help="LR image names.")

parser.add_argument("--dataset", type=str, choice=("cd", "rcc", "luad"), help="Dataset to choose.")
parser.add_argument("--gpu_id", type=int, default=0, help="Index of GPU to run the model on.")

parser.add_argument("--resize_mode", type=str, default="maxpool+bicubic",
                    choice=("maxpool", "bicubic", "maxpool+bicubic"),
                    help="Resizing functions for matching the size of teacher and student outputs.")

params = parser.parse_args()
