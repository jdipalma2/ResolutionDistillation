import argparse
import pickle
from pathlib import Path

import numpy as np
import sklearn.metrics as metrics
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from model.resnet_with_groupnorm import ResNet
from utils import compute_ci

parser = argparse.ArgumentParser(description="Compute Confidence Intervals", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--num_iter", type=int, default=10000, help="Number of times to evaluate model.")
parser.add_argument("--num_samples", type=int, help="Number of samples to select from the dataset each iteration.")
parser.add_argument("--confidence", type=float, default=0.95, help="Confidence interval value.")
parser.add_argument("--model_ckpt", type=Path, help="Trained model checkpoint.")
parser.add_argument("--data_dir", type=Path, help="Data to compute metrics on.")
parser.add_argument("--num_workers", type=int, default=10, help="Number of sub-processes for IO.")
#parser.add_argument("--average", type=str, default="macro", help="Type of averaging to use for computing aggregated metrics.")

args = parser.parse_args()

print(f"##########     CONFIGURATION     ##########\n")
print(f"{chr(10).join(f'{k}:{chr(9)}{v}' for k, v in vars(args).items())}\n")

class_names = ["Abnormal", "Normal", "Sprue"]
class_nums = list(range(len(class_names)))

with open("image_stats.pickle", "rb") as f:
    mean, std = pickle.load(f)


print(f"class_names:\t{class_names}")
print(f"class_nums:\t{class_nums}")
print(f"mean:\t{mean}")
print(f"std:\t{std}")
print(f"\n###########################################\n\n\n")




with torch.no_grad():

    transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    class_counts = [0 for __ in range(len(class_names))]
    ds = datasets.ImageFolder(root=args.data_dir, transform=transform)
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    y_true = torch.empty(size=(len(ds), ), dtype=torch.long).cpu()
    y_pred = torch.empty(size=(len(ds), ), dtype=torch.long).cpu()

    model = ResNet(planes=(32, 64, 128, 256), layers=(1, 1, 1, 1), num_classes=len(class_names))
    model.load_state_dict(torch.load(f=args.model_ckpt)["student_model_state_dict"])
    model = model.cuda()
    model = model.train(mode=False)

    class_counts = [0 for __ in range(len(class_names))]
    for idx, (data, label) in enumerate(tqdm(dl, desc="Inference")):
        class_counts[label.detach().cpu().item()] += 1
        y_true[idx] = label.detach().cpu()
        y_pred[idx] = torch.max(model(data.cuda(non_blocking=True))[-1], dim=1)[1].detach().cpu()

    del model
    torch.cuda.empty_cache()

    weights = []
    for i in range(len(y_true)):
        weights.append(1.0 / class_counts[y_true[i].item()])

    accs = [[] for __ in range(len(class_names) + 1)]
    f1s = [[] for __ in range(len(class_names) + 1)]
    pres = [[] for __ in range(len(class_names) + 1)]
    recs = [[] for __ in range(len(class_names) + 1)]

    for n in tqdm(range(args.num_iter), desc="Iteration"):
        random_samples = list(torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=args.num_samples, replacement=True))

        y_true_rs = y_true[random_samples]
        y_pred_rs = y_pred[random_samples]

        # Per-class accuracy computation from Naofumi Tomita.
        cm = torch.zeros(size=(len(class_names), len(class_names)), dtype=torch.float).cpu()
        for t in range(y_pred_rs.numel()):
            p = y_pred_rs[t].item()
            g = y_true_rs[t].item()
            cm[p, g] += 1

        f = metrics.f1_score(y_true=y_true_rs, y_pred=y_pred_rs, labels=class_nums, average=None, zero_division=0)
        p = metrics.precision_score(y_true=y_true_rs, y_pred=y_pred_rs, labels=class_nums, average=None, zero_division=0)
        r = metrics.recall_score(y_true=y_true_rs, y_pred=y_pred_rs, labels=class_nums, average=None, zero_division=0)

        for c in range(len(class_names)):
            f1s[c].append(f[c])
            pres[c].append(p[c])
            recs[c].append(r[c])

            tp = cm[c, c].item()
            tn = cm.sum().item() - tp
            fp = cm[c, :].sum().item() - tp
            fn = cm[:, c].sum().item() - tp
            pos = tp + fn
            neg = tn + fp
            accs[c].append((tp + tn) / (pos + neg))

        accs[-1].append((accs[0][-1] + accs[1][-1] + accs[2][-1]) / 3)
        f1s[-1].append((f1s[0][-1] + f1s[1][-1] + f1s[2][-1]) / 3)
        pres[-1].append((pres[0][-1] + pres[1][-1] + pres[2][-1]) / 3)
        recs[-1].append((recs[0][-1] + recs[1][-1] + recs[2][-1]) / 3)

    # Compute the confidence intervals and print.
    for idx, cn in enumerate(class_names):
        print(f"Class Name:\t{cn}")
        lower, upper = compute_ci(arr=accs[idx], ci=args.confidence)
        print(f"Accuracy:\t{np.mean(accs[idx]):.2f} ({lower:.2f}-{upper:.2f})")
        lower, upper = compute_ci(arr=f1s[idx], ci=args.confidence)
        print(f"F1-Score:\t{np.mean(f1s[idx]):.2f} ({lower:.2f}-{upper:.2f})")
        lower, upper = compute_ci(arr=pres[idx], ci=args.confidence)
        print(f"Precision:\t{np.mean(pres[idx]):.2f} ({lower:.2f}-{upper:.2f})")
        lower, upper = compute_ci(arr=recs[idx], ci=args.confidence)
        print(f"Recall:\t{np.mean(recs[idx]):.2f} ({lower:.2f}-{upper:.2f})")
        print("\n\n\n")


    print(f"Means:")
    idx = -1
    lower, upper = compute_ci(arr=accs[idx], ci=args.confidence)
    print(f"Accuracy:\t{np.mean(accs[idx]):.4f} ({lower:.4f}-{upper:.4f})")
    lower, upper = compute_ci(arr=f1s[idx], ci=args.confidence)
    print(f"F1-Score:\t{np.mean(f1s[idx]):.4f} ({lower:.4f}-{upper:.4f})")
    lower, upper = compute_ci(arr=pres[idx], ci=args.confidence)
    print(f"Precision:\t{np.mean(pres[idx]):.4f} ({lower:.4f}-{upper:.4f})")
    lower, upper = compute_ci(arr=recs[idx], ci=args.confidence)
    print(f"Recall:\t{np.mean(recs[idx]):.4f} ({lower:.4f}-{upper:.4f})")

