import datetime
import pickle

import pandas as pd
import torch
from tqdm import tqdm

from config import params
from datasets.factory import create_dataloader, create_transforms
from model.factory import KD
from training.loss import kd_loss, pixel_loss
from training.run_epoch import run_epoch


def main(args):
    # Set up various configuration details.
    # CUDA device to run on (GPU required)
    device = torch.device(f"cuda:{args.gpu_id}")
    # Select the loss function.
    loss_func = torch.nn.L1Loss() if args.loss_type == "l1" else torch.nn.MSELoss()
    # Number of classes changes per dataset.
    num_classes = 5 if args.dataset == "luad" else 3
    # Find per-channel mean and standard deviation
    with open(f"{args.dataset}_image_stats.pickle", "rb") as f:
        mean, std = pickle.load(f)
    # Modify the data directory
    args.data_dir = args.data_dir.joinpath(f"data_{args.ds_ext}")

    # Set up logging
    # Create directories if needed
    args.log_dir.mkdir(parents=True, exist_ok=True)
    # Create identifier based on time for logging
    # (making bad assumption that only 1 run will be started at this exact time)
    now = datetime.datetime.now()
    log_name = f"{now.month}{now.day}{now.year}_{now.hour}{now.minute}{now.second}_{now.microsecond}"
    args.checkpoint_dir.joinpath(log_name).mkdir(exist_ok=True, parents=True)

    # Create data transforms
    train_transforms, val_transforms = create_transforms(args=args, mean=mean, std=std)
    # Create datasets
    train_dl = create_dataloader(data_dir=args.data_dir.joinpath("train"), transform=train_transforms,
                                 ds_ext=args.ds_ext, num_workers=args.num_workers, shuffle=True)
    val_dl = create_dataloader(data_dir=args.data_dir.joinpath("val"), transform=val_transforms, ds_ext=args.ds_ext,
                               num_workers=args.num_workers, shuffle=False)

    # Create the model
    model = KD(num_classes=num_classes, teacher_saved_model=args.teacher_saved_model)

    # Create the optimizer and learning rate scheduler.
    optimizer = torch.nn.optim.Adam(params=model.student_model.parameters(), lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
    scheduler = torch.nn.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)

    train_loss = []
    val_loss = []

    # TODO Write code to save hyper-parameters to log file

    for epoch in tqdm(range(args.num_epochs)):
        print(10 * "#" + 5 * " " + f"Running epoch {epoch}" + 5 * " " + 10 * "#" + "\n\n")

        # Send models to GPU
        model.student_model.to(device)
        model.teacher_model.to(device)

        # Training
        model.student_model.train()
        train_loss.append(
            run_epoch(dl=train_dl, model=model, optimizer=optimizer, is_train=True, device=device, soft_loss=kd_loss,
                      pixel_loss=pixel_loss, loss_func=loss_func, temperature=args.temperature,
                      batch_size=args.batch_size, resize_mode=args.resize_mode))

        # Validation
        model.student_model.eval()
        with torch.no_grad():
            val_loss.append(run_epoch(dl=val_dl, model=model, is_train=False, device=device, soft_loss=kd_loss,
                                      pixel_loss=pixel_loss, loss_func=loss_func, temperature=args.temperature,
                                      batch_size=args.batch_size, resize_mode=args.resize_mode))

        print(f"Training loss:\t{train_loss[-1]:.4f}\t\tValidation loss:\t{val_loss[-1]:.4f}\n")

        scheduler.step()

        # Save checkpoints
        # Transfer models to CPU for ease of loading
        model.student_model.cpu()
        model.teacher_model.cpu()
        torch.save({"student_model_state_dict": model.student_model.state_dict(),
                    "teacher_model_state_dict": model.teacher_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()},
                   f=args.checkpoint_dir.joinpath(f"epoch_{epoch}.pt"))

    # Save the training metrics
    df = pd.DataFrame()
    df["Epoch"] = list(range(args.num_epochs))
    df["Training Loss"] = train_loss
    df["Validation Loss"] = val_loss
    df.to_csv(args.log_dir.joinpath(f"log_{log_name}.csv"), index=False)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This code requires a CUDA-compatible GPU to run."

    main(params)
