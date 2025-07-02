import argparse
import datetime
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model.resnet_with_groupnorm import ResNet
from utils import calculate_confusion_matrix, compute_auc_multi, save_confusion_matrix, save_roc_curve_multi

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description="Knowledge Distillation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--num_workers", type=int, default=10, help="Number of subprocesses for IO.")
parser.add_argument("--num_epochs", type=int, default=100,
                    help="Number of model checkpoints (how many epochs the model was trained for).")
parser.add_argument("--data_dir", type=Path, help="Location of training and validation datasets.")
parser.add_argument("--checkpoint_dir", type=Path, help="Path where the model checkpoints are stored.")
parser.add_argument("--log_dir", type=Path, default=Path("logs_test_results"),
                    help="Path to store training and validation metrics to CSV.")
parser.add_argument("--roc_dir", type=Path, default=Path("roc_test_results"), help="Path to store ROC curve plots to.")
parser.add_argument("--cm_dir", type=Path, default=Path("cm_test_results"), help="Path to store confusion matrices to.")

parser.add_argument("--dataset", type=str, choice=("cd", "luad", "rcc"),
                    help="Dataset to evaluate on. Only 3 officially supported.")
args = parser.parse_args()

# Dataset specific configuration
# Class names
if args.dataset == "cd":
    class_names = ("Abnormal", "Normal", "Sprue")
elif args.dataset == "luad":
    class_names = ("Acinar", "Lepidic", "Micropapillary", "Papillary", "Solid")
elif args.dataset == "rcc":
    class_names = ("Clearcell", "Chromophobe", "Papillary")
else:
    # Shouldn't reach this point as the argparse shouldn't allow invalid choices
    raise NotImplementedError(f"Invalid selection for dataset ({args.dataset})")
# Per-channel statistics
with open(f"{args.dataset}_image_stats.pickle", "rb") as f:
    mean, std = pickle.load(f)

# Make sure the log directory exists.
args.log_dir.mkdir(parents=True, exist_ok=True)

now = datetime.datetime.now()
log_name = f"{now.month}{now.day}{now.year}_{now.hour}{now.minute}{now.second}"
log_csv = args.log_dir.joinpath(f"log_{log_name}.csv")

# Make sure the output directories exist.
args.roc_dir.joinpath(log_name).mkdir(parents=True, exist_ok=True)
args.cm_dir.joinpath(log_name).mkdir(parents=True, exist_ok=True)

# Sets to evaluate.
sets = ("train", "val", "test")

# Loss function
loss_func = torch.nn.CrossEntropyLoss()

# Create the model.
model = ResNet(planes=(32, 64, 128, 256), layers=(1, 1, 1, 1), num_classes=len(class_names)).cuda()

# Create the datasets transforms.
data_transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

# Create the datasets.
dataset = {s: datasets.ImageFolder(root=args.data_dir.joinpath(s), transform=data_transform) for s in sets}

# Create the datasets loaders.
loader = {s: torch.utils.data.DataLoader(dataset=dataset[s], batch_size=1, shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True) for s in sets}

# Print the configuration.
print(
    f"##########      CONFIGURATION     ##########\n{chr(10).join(f'{k}:{chr(9)}{v}' for k, v in vars(args).items())}\n")
print(f"mean:\t{mean}\n")
print(f"standard deviation:\t{std}\n")
print(f"log_name:\t{log_name}\n")
print(f"log_csv:\t{log_csv}\n")
print(f"sets:\t{sets}\n")
print(f"classes:\t{classes}\n")
print("\n###########################################\n\n")
print("\n\n##########     DATA TRANSFORMS     ##########\n")
print(f"datasets transform:\t{data_transform}\n")
print("\n#############################################\n\n")
print("\n\n##########     DATASET     ##########\n")
print(f"Number of training images:\t{len(dataset['train'])}")
print(f"Number of validation images:\t{len(dataset['val'])}")
print(f"Number of test images:\t{len(dataset['test'])}")
print("\n#####################################\n\n")


def find_results(model, dataloader, num_classes):
    with torch.no_grad():
        all_labels = torch.empty(size=(len(dataloader),), dtype=torch.long).cpu()
        all_predicts = torch.empty(size=(len(dataloader), num_classes), dtype=torch.float).cpu()

        for idx, (image, label) in enumerate(dataloader):
            all_labels[idx] = label.detach().cpu()
            all_predicts[idx] = model(image.cuda(non_blocking=True))[-1].detach().cpu()

    return all_labels, all_predicts


with log_csv.open(mode="w") as writer:
    writer.write(
        "Epoch,Training Loss,Training Accuracy,Training Abnormal AUC,Training Normal AUC,Training Sprue AUC,Validation Loss,Validation Accuracy,Validation Abnormal AUC,Validation Normal AUC,Validation Sprue AUC,Test Loss,Test Accuracy,Test Abnormal AUC,Test Normal AUC,Test Sprue AUC\n")

    for epoch in tqdm(range(args.num_epochs), desc="Epoch"):
        model.load_state_dict(
            torch.load(f=args.checkpoint_dir.joinpath(f"epoch_{epoch}.pt"))["student_model_state_dict"], strict=True)
        model.train(mode=False)

        train_all_labels, train_all_predicts = find_results(model=model, dataloader=loader["train"])
        val_all_labels, val_all_predicts = find_results(model=model, dataloader=loader["val"])
        test_all_labels, test_all_predicts = find_results(model=model, dataloader=loader["test"])

        train_loss = nn.CrossEntropyLoss()(input=train_all_predicts, target=train_all_labels).item()
        train_accuracy = torch.mean(
            (torch.max(train_all_predicts, dim=1)[1] == train_all_labels).to(torch.float32)).item()
        val_loss = nn.CrossEntropyLoss()(input=val_all_predicts, target=val_all_labels).item()
        val_accuracy = torch.mean((torch.max(val_all_predicts, dim=1)[1] == val_all_labels).to(torch.float32)).item()
        test_loss = nn.CrossEntropyLoss()(input=test_all_predicts, target=test_all_labels).item()
        test_accuracy = torch.mean((torch.max(test_all_predicts, dim=1)[1] == test_all_labels).to(torch.float32)).item()

        train_cm = calculate_confusion_matrix(all_labels=train_all_labels.numpy(),
                                              all_predicts=torch.max(train_all_predicts, dim=1)[1].numpy(),
                                              classes=class_names)
        val_cm = calculate_confusion_matrix(all_labels=val_all_labels.numpy(),
                                            all_predicts=torch.max(val_all_predicts, dim=1)[1].numpy(),
                                            classes=class_names)
        test_cm = calculate_confusion_matrix(all_labels=test_all_labels.numpy(),
                                             all_predicts=torch.max(test_all_predicts, dim=1)[1].numpy(),
                                             classes=class_names)

        # Save the ROC curves.
        __ = save_roc_curve_multi(obs_lists=torch.nn.functional.one_hot(train_all_labels).tolist(),
                                  pred_lists=train_all_predicts.tolist(),
                                  figdest=str(args.roc_dir.joinpath(log_name).joinpath(f"train_epoch_{epoch}.png")),
                                  class_names=class_names, show_micro_avg=True, show_macro_avg=True)
        __ = save_roc_curve_multi(obs_lists=torch.nn.functional.one_hot(val_all_labels).tolist(),
                                  pred_lists=val_all_predicts.tolist(),
                                  figdest=str(args.roc_dir.joinpath(log_name).joinpath(f"val_epoch_{epoch}.png")),
                                  class_names=class_names, show_micro_avg=True, show_macro_avg=True)
        __ = save_roc_curve_multi(obs_lists=torch.nn.functional.one_hot(test_all_labels).tolist(),
                                  pred_lists=test_all_predicts.tolist(),
                                  figdest=str(args.roc_dir.joinpath(log_name).joinpath(f"test_epoch_{epoch}.png")),
                                  class_names=class_names, show_micro_avg=True, show_macro_avg=True)

        # Compute the AUCs.
        train_auc = compute_auc_multi(labels=torch.nn.functional.one_hot(train_all_labels).tolist(),
                                      preds=train_all_predicts.tolist())
        val_auc = compute_auc_multi(labels=torch.nn.functional.one_hot(val_all_labels).tolist(),
                                    preds=val_all_predicts.tolist())
        test_auc = compute_auc_multi(labels=torch.nn.functional.one_hot(test_all_labels).tolist(),
                                     preds=test_all_predicts.tolist())

        # Save the confusion matrices.
        save_confusion_matrix(cm=train_cm, class_names=class_names,
                              output_name=args.cm_dir.joinpath(log_name).joinpath(f"train_epoch_{epoch}.png"))
        save_confusion_matrix(cm=val_cm, class_names=class_names,
                              output_name=args.cm_dir.joinpath(log_name).joinpath(f"val_epoch_{epoch}.png"))
        save_confusion_matrix(cm=test_cm, class_names=class_names,
                              output_name=args.cm_dir.joinpath(log_name).joinpath(f"test_epoch_{epoch}.png"))

        writer.write(f"{epoch},"
                     f"{train_loss},"
                     f"{train_accuracy},"
                     f"{train_auc_a},"
                     f"{train_auc_n},"
                     f"{train_auc_s},"
                     f"{val_loss},"
                     f"{val_accuracy},"
                     f"{val_auc_a},"
                     f"{val_auc_n},"
                     f"{val_auc_s},"
                     f"{test_loss},"
                     f"{test_accuracy},"
                     f"{test_auc_a},"
                     f"{test_auc_n},"
                     f"{test_auc_s}\n")
