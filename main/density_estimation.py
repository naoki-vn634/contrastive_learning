import argparse
import json
import os
import sys
import shutil
from distutils.util import strtobool
from glob import glob
from math import log2

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


def calculate_entropy(pos):
    entropy = np.array([])

    for posterior in pos:
        ent = -sum([x * log2(x) if x != 0 else 0 for x in posterior])
        entropy = np.append(entropy, ent)
    return entropy


def get_ood_rank(x, test_score):
    ind = np.where(test_score > x)[0]
    if not len(ind) == 0:
        ood_rank = len(test_score) - ind[0]
    else:
        ood_rank = 0
    ood_rank /= len(test_score)
    return ood_rank


def save_score(args, label_dict, score_dict, ano_dict, ranch):
    phase_list = ["val"]
    # phase_list = ["test", "ood"]
    plt.figure()
    for phase in phase_list:
        # print(np.max(score_dict[phase]))
        sns.distplot(score_dict[phase], label=phase, hist=False)
        print("max ", np.max(score_dict[phase]))
        print("min ", np.min(score_dict[phase]))
    plt.xlim([-100000, 0])
    plt.legend()
    plt.savefig(os.path.join(args.output, "score_garbage.png"))
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
    ax1.set_xlim(args.thu, 0)
    ax2.set_xlim(args.thu, 0)
    id = ["train", "test", "val"]
    ood = ["ood", "garbage"]

    for phase in id:
        extract = score_dict[phase][np.where(score_dict[phase] > args.thu)[0]]
        sns.distplot(extract, label=phase, hist=True, ax=ax1, bins=100)
    for phase in ood:
        extract = score_dict[phase][np.where(score_dict[phase] > args.thu)[0]]
        extract_ano = ano_dict[phase][np.where(score_dict[phase] > args.thu)[0]]
        if phase == "garbage":
            for i in range(len(ranch)):
                extract_ranch = extract[np.where(extract_ano == i)[0]]
                print(extract_ranch)
                sns.distplot(extract_ranch, label=ranch[i], hist=True, ax=ax2)
        else:
            sns.distplot(extract, label=phase, hist=True, ax=ax2, bins=100)
    ax1.set_xlabel("ood score s(x)")
    ax2.set_xlabel("ood score s(x)")
    fig.suptitle("ood score comparison (Epoch: {})".format(args.epoch), fontsize=16)
    ax1.legend()
    ax2.legend()

    fig.savefig(os.path.join(args.output, "score_subplot.png"))
    plt.close()


def evaluate(args, net, dataloaders_dict, device, extract_path, ranch):
    phase_list = ["train", "test", "val", "ood", "garbage"]
    torch.backends.cudnn.benchmark = True
    label_dict = dict()
    ano_dict = dict()
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

            if args.hidden == 0 or 1:
                features = middle if i == 0 else torch.cat([features, middle], dim=0)
            # elif args.hidden == 1:
            #     features = out if i == 0 else torch.cat([features, out], dim=0)
            labels = label if i == 0 else torch.cat([labels, label], dim=0)
            poses = pos if i == 0 else torch.cat([poses, pos], dim=0)
            # if phase == "ood":
            #     all_image = (
            #         image.cpu().data.numpy()
            #         if i == 0
            #         else np.concatenate((all_image, image.cpu().data.numpy()), axis=0)
            #     )
        _, preds = torch.max(poses, 1)

        if phase == "train":
            # preds_np = preds.cpu().data.numpy()
            # labels_np = labels.cpu().data.numpy()
            # features_np = features.cpu().data.numpy()
            # poses_np = poses.cpu().data.numpy()
            # ent_np = calculate_entropy(poses_np)
            # print(len(features_np))
            # features_ex = torch.from_numpy(features_np[np.where((labels_np == preds_np) & (ent_np < 0.01))[0]]).to(device)
            # labels_ex = torch.from_numpy(labels_np[np.where((labels_np == preds_np) & (ent_np < 0.01))[0]]).to(device)
            # print(len(features_ex))
            # cov_class, mean_class = compute_covar_mean(args, features_ex, labels_ex)
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
        ano_dict[phase] = labels.cpu().data.numpy()
        # np.save(os.path.join(args.output, f"{phase}_pos.npy"), poses.cpu().data.numpy())
        # np.save(
        #     os.path.join(args.output, f"{phase}_label.npy"), labels.cpu().data.numpy()
        # )
        # np.save(os.path.join(args.output, f"{phase}_score.npy"), score)

    # score_extract = score_dict["ood"]
    # print(len(score_extract))
    # for image, score in zip(all_image, score_extract):
    #     image = np.transpose(image, (1, 2, 0)) * 255

    #     far_dir = os.path.join(args.output, "far_id_ood")
    #     near_dir = os.path.join(args.output, "near_id_ood")
    #     if score > -2500:  # id
    #         save_name = os.path.join(near_dir, f"{str(score)}_ood.jpg")
    #     elif score <= 2500:
    #         save_name = os.path.join(far_dir, f"{str(score)}_ood.jpg")
    #     cv2.imwrite(save_name, image)
    save_score(args, label_dict, score_dict, ano_dict, ranch)

    return score_dict, label_dict


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)
    phase_list = ["test", "val", "ood", "garbage"]
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Data load
    with open(os.path.join(args.id, "data.json")) as f:
        data = json.load(f)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    if args.ranch:
        val_path = glob(
            "/mnt/aoni02/matsunaga/Dataset/200313_global-model-ranch/garbage_ver3/ood_ranch/*/*/*"
        )

    else:
        val_path = glob("/mnt/aoni02/matsunaga/10_cropped-images/all_id/*/*.jpg")
    val_label = [2 for _ in range(len(val_path))]

    ood_path = glob(os.path.join(args.ood, "*/*.jpg"))
    ood_label = [3 for _ in range(len(ood_path))]

    if args.ranch:
        garbage_dir = glob(
            "/mnt/aoni02/matsunaga/Dataset/200313_global-model-ranch/garbage_ver3/train/garbage/*"
        )
        ranch = [os.path.basename(dir) for dir in garbage_dir]
        print(ranch)
        garbage_path = []
        garbage_label = []
        for i, dir in enumerate(garbage_dir):
            img = glob(os.path.join(dir, "*"))
            garbage_path.extend(img)
            garbage_label.extend([i for _ in range(len(img))])
    else:
        garbage_path = glob(
            "/mnt/aoni02/matsunaga/Dataset/200313_global-model/garbage_ver3/train/garbage/*"
        )
        garbage_label = [4 for _ in range(len(garbage_path))]

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

    garbage_dataset = ImageDataset(
        garbage_path, garbage_label, transform=transforms, phase="val"
    )
    garbage_dataloader = torch.utils.data.DataLoader(
        garbage_dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )

    dataloaders_dict = {
        "train": train_dataloader,
        "test": test_dataloader,
        "val": val_dataloader,
        "ood": ood_dataloader,
        "garbage": garbage_dataloader,
    }
    print("train: ", len(train_dataloader.dataset))
    print("test: ", len(test_dataloader.dataset))
    print("val: ", len(val_dataloader.dataset))
    print("ood: ", len(ood_dataloader.dataset))
    print("garbage: ", len(garbage_dataloader.dataset))

    ood_rank = dict()
    score_dict, label_dict = evaluate(
        args, net, dataloaders_dict, device=device, extract_path=ood_path, ranch=ranch
    )
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
    parser.add_argument("--epoch", type=str)
    parser.add_argument("--ranch", type=strtobool)
    parser.add_argument("--id", type=str)
    parser.add_argument("--ood", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--hidden", type=int, help="0:2048, 1:128")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_cls", type=int, default=2)
    parser.add_argument("--thu", type=int)
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
