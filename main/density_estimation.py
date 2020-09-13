import argparse
import json
import os
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
from utils import compute_covar_mean, density_score
from loss_func import ContrastiveLoss

sys.path.append("../models/")
from model import ContrastiveResnetModel


def get_ood_rank(x, test_score):
    ind = np.where(test_score > x)[0]
    if not len(ind) == 0:
        ood_rank = len(test_score) - ind[0]
    else:
        ood_rank = 0
    ood_rank /= len(test_score)
    return ood_rank


def save_score(args, label_dict, score_dict):
    phase_list = ["train", "test", "val", "ood"]
    plt.figure()
    for phase in phase_list:
        # print(np.max(score_dict[phase]))
        sns.distplot(score_dict[phase], label=phase, hist=False)
    if args.hidden == 0:
        plt.xlim([-10000, 0])
    plt.legend()
    plt.savefig(os.path.join(args.output, "score.png"))


def evaluate(args, net, dataloaders_dict, device):
    phase_list = ["train", "test", "val", "ood"]
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
            pos = nn.functional.softmax(net.fc3(middle), dim=1)

            if args.hidden == 0:
                features = middle if i == 0 else torch.cat([features, middle], dim=0)
            elif args.hidden == 1:
                features = out if i == 0 else torch.cat([features, out], dim=0)
            labels = label if i == 0 else torch.cat([labels, label], dim=0)
            poses = pos if i == 0 else torch.cat([poses, pos], dim=0)

        if phase == "train":
            cov_class, mean_class = compute_covar_mean(args, features, labels)

        ind, score, posterior = np.array([]), np.array([]), np.array([])
        for feature in features:
            score_label, scores = density_score(
                args, feature, cov_class, mean_class, device
            )

            score = np.append(score, float(scores))
            ind = np.append(ind, int(score_label))
        label_dict[phase] = ind
        score_dict[phase] = score

        np.save(os.path.join(args.output, f"{phase}_pos.npy"), poses.cpu().data.numpy())
        np.save(
            os.path.join(args.output, f"{phase}_label.npy"), labels.cpu().data.numpy()
        )
        np.save(os.path.join(args.output, f"{phase}_score.npy"), score)

    save_score(args, label_dict, score_dict)

    return score_dict, label_dict


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)
    phase_list = ["test", "val", "ood"]
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Data load
    with open(os.path.join(args.id, "data.json")) as f:
        data = json.load(f)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    val_path = glob("/mnt/aoni02/matsunaga/10_cropped-images/all_id/*/*.jpg")
    val_label = [2 for _ in range(len(val_path))]

    ood_path = glob(os.path.join(args.ood, "*/*.jpg"))
    ood_label = [3 for _ in range(len(ood_path))]

    transforms = ImageTransform(batchsize=args.batchsize)

    net = ContrastiveResnetModel(out_dim=args.n_cls, hidden=args.hidden)
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

    val_dataset = ImageDataset(val_path, val_label, transform=transforms, phase="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )

    ood_dataset = ImageDataset(ood_path, ood_label, transform=transforms, phase="val")
    ood_dataloader = torch.utils.data.DataLoader(
        ood_dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )

    dataloaders_dict = {
        "train": train_dataloader,
        "test": test_dataloader,
        "val": val_dataloader,
        "ood": ood_dataloader,
    }

    ood_rank = dict()
    score_dict, label_dict = evaluate(args, net, dataloaders_dict, device=device)
    for phase in phase_list:
        if phase == "test":
            test_score = score_dict[phase]
            test_score.sort()

        ood_rank[phase] = 1 - np.mean(
            [get_ood_rank(x, test_score) for x in score_dict[phase]]
        )
        print(ood_rank)
    with open(os.path.join(args.output, "ood_rank.json"), "w") as f:
        d = json.dumps(ood_rank)
        f.write(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str)
    parser.add_argument("--ood", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--hidden", type=int, help="0:2048, 1:128")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_cls", type=int, default=2)
    parser.add_argument(
        "--gpuid",
        type=str,
    )

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main(args)
