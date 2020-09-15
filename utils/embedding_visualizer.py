import argparse
import os
import torch
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from distutils.util import strtobool
from glob import glob
from sklearn.manifold import TSNE


sys.path.append("../preprocess/")
from img_preprocess import ImageDataset, ImageTransform

sys.path.append("../utils/")
from loss_func import ContrastiveLoss

sys.path.append("../models/")
from model import ContrastiveResnetModel


def save_id_ood(
    args, tsne_train, tsne_test, tsne_val, train_label, test_label, val_label
):
    cm = plt.cm.get_cmap("tab20")
    plt.figure()
    for i in range(args.n_cls):
        emb_train = tsne_train[np.where(train_label == i)[0]]
        emb_test = tsne_test[np.where(test_label == i)[0]]
        plt.scatter(
            emb_train[:, 0],
            emb_train[:, 1],
            label="train",
            marker="*",
            color=cm.colors[i],
            alpha=0.2,
        )
        plt.scatter(
            emb_test[:, 0],
            emb_test[:, 1],
            label="test",
            marker="o",
            color=cm.colors[i],
            alpha=0.2,
        )
    plt.scatter(
        tsne_val[:, 0], tsne_val[:, 1], label="ood", color="blue", marker="v", alpha=0.2
    )
    train_marker = mlines.Line2D(
        [], [], marker="*", linestyle=None, markersize=10, label="train"
    )
    test_marker = mlines.Line2D(
        [], [], marker="o", linestyle=None, markersize=10, label="test"
    )
    val_marker = mlines.Line2D(
        [], [], marker="v", linestyle=None, markersize=10, label="ood"
    )

    plt.title("embedding space")
    plt.legend(handles=[train_marker, test_marker, val_marker])
    plt.savefig(os.path.join(args.output, "id_ood_embedding.png"))


def embedder(dataloader, net, device):
    for ind, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        _, emb = net(image)
        embedding = (
            emb.cpu().data.numpy()
            if ind == 0
            else np.concatenate([embedding, emb.cpu().data.numpy()], axis=0)
        )
        labels = (
            label.cpu().data.numpy()
            if ind == 0
            else np.concatenate([labels, label.cpu().data.numpy()], axis=0)
        )
    return embedding, labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)

    with open(os.path.join(args.input, "data.json")) as f:
        data = json.load(f)
        x_train, x_test = data["x_train"], data["x_test"]
        y_train, y_test = data["y_train"], data["y_test"]

    ood_img = glob("/mnt/aoni02/matsunaga/ImageNet/cow_resemble/*/*.jpg")
    ood_label = [2 for _ in range(len(ood_img))]

    net = ContrastiveResnetModel(out_dim=args.n_cls)
    net.to(device)
    net.load_state_dict(torch.load(args.weight))

    transforms = ImageTransform(batchsize=args.batchsize)
    train_dataset = ImageDataset(x_train, y_train, transform=transforms, phase="val")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, num_workers=40, shuffle=True
    )

    test_dataset = ImageDataset(x_test, y_test, transform=transforms, phase="val")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, num_workers=40, shuffle=False
    )

    val_dataset = ImageDataset(ood_img, ood_label, transform=transforms, phase="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, num_workers=40, shuffle=False
    )
    print("embedding")
    train_emb, train_label = embedder(train_dataloader, net, device)
    test_emb, test_label = embedder(test_dataloader, net, device)
    val_emb, val_label = embedder(val_dataloader, net, device)
    all_emb = np.concatenate([train_emb, test_emb, val_emb], axis=0)
    if args.tsne:
        print("tsne")
        tsne = TSNE(n_components=2).fit_transform(all_emb)
        tsne_train = tsne[: len(train_label)]
        tsne_test = tsne[len(train_label) : len(train_label) + len(test_label)]
        tsne_val = tsne[
            len(train_label)
            + len(test_label) : len(train_label)
            + len(test_label)
            + len(val_label)
        ]

        save_id_ood(
            args, tsne_train, tsne_test, tsne_val, train_label, test_label, val_label
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_cls", type=int)
    parser.add_argument("--gpuid", type=str)
    parser.add_argument('--tsne', type=strtobool)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main(args)
