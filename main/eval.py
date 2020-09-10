import argparse
import json
import os
import random
import sys
from distutils.util import strtobool
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../preprocess/")
from img_preprocess import ImageDataset, ImageTransform

sys.path.append("../utils/")
from density_estimation import compute_covar_mean, density_score
from loss_func import ContrastiveLoss

sys.path.append("../models/")
from model import ContrastiveResnetModel


def save_score(args, label_dict, score_dict):
    phase_list = ["train", "test", "val"]
    plt.figure()
    for phase in phase_list:
        print(np.max(score_dict[phase]))
        sns.distplot(score_dict[phase], label=phase)
    plt.xlim([-10000, 0])
    plt.legend()
    plt.savefig(os.path.join(args.output, "score.png"))


def evaluate(args, net, dataloaders_dict, device):
    phase_list = ["train", "test", "val"]
    torch.backends.cudnn.benchmark = True
    label_dict = dict()
    score_dict = dict()

    for phase in phase_list:
        print("##Phase: ", phase)
        net.eval()
        torch.set_grad_enabled(False)

        for i, (image, label) in enumerate(dataloaders_dict[phase]):
            image = image.to(device)
            label = label.to(device)
            out, middle = net(image)

            features = middle if i == 0 else torch.cat([features, middle], dim=0)
            labels = label if i == 0 else torch.cat([labels, label], dim=0)
        if phase == "train":
            cov_class, mean_class = compute_covar_mean(args, features, labels)

        ind, score = np.array([]), np.array([])
        for feature in features:
            score_label, scores = density_score(
                args, feature, cov_class, mean_class, device
            )

            score = np.append(score, float(scores))
            ind = np.append(ind, int(score_label))
        label_dict[phase] = ind
        score_dict[phase] = score
    save_score(args, label_dict, score_dict)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Data load
    with open(os.path.join(args.id, "data.json")) as f:
        data = json.load(f)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    ood_path = glob(os.path.join(args.ood, "*/*.jpg"))
    ood_label = [4 for _ in range(len(ood_path))]

    transforms = ImageTransform(batchsize=args.batchsize)

    net = ContrastiveResnetModel(out_dim=args.n_cls)
    net.to(device)
    net.load_state_dict(torch.load(args.weight))

    train_dataset = ImageDataset(x_train, y_train, transform=transforms, phase="val")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, num_workers=1, shuffle=True
    )

    test_dataset = ImageDataset(x_test, y_test, transform=transforms, phase="val")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )

    val_dataset = ImageDataset(ood_path, ood_label, transform=transforms, phase="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )

    dataloaders_dict = {
        "train": train_dataloader,
        "test": test_dataloader,
        "val": val_dataloader,
    }

    evaluate(args, net, dataloaders_dict, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str)
    parser.add_argument("--ood", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_cls", type=int)
    parser.add_argument("--gpuid", type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main(args)
