import argparse
import cv2
import json
import os
import random
import sys
from distutils.util import strtobool
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../preprocess/")
from img_preprocess import ImageDataset, ImageTransform

sys.path.append("../utils/")
from loss_func import ContrastiveLoss

sys.path.append("../models/")
from model import ContrastiveResnetModel


def save_embedding(args, emb_0, emb_1, emb_2, tblogger, epoch, phase, save_dir):
    plt.figure()
    plt.scatter(emb_0[:, 0], emb_0[:, 1], color="red", label="no", alpha=0.5)
    plt.scatter(emb_1[:, 0], emb_1[:, 1], color="blue", label="yes", alpha=0.5)
    plt.scatter(emb_2[:, 0], emb_2[:, 1], color="green", label="garbage", alpha=0.5)
    plt.title(f"Epoch {epoch} Embedding Space")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{epoch}_{phase}_embedding.png"))
    plt.close()
    emb_img = cv2.imread(os.path.join(save_dir, f"{epoch}_{phase}_embedding.png"))
    emb_img = np.transpose(emb_img, (2, 0, 1))
    tblogger.add_image(f"embedding_{phase}/{epoch}", emb_img)


def trainer(args, net, dataloaders_dict, output, optimizer, criterion, device, tfboard):
    emb_save_dir = os.path.join(args.output, "embedding")
    weight_save_dir = os.path.join(args.output, "weight")
    for dir in [emb_save_dir, weight_save_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    phase_list = ["train", "test"]
    Loss = {"train": [0] * args.epoch, "test": [0] * args.epoch}
    # Acc = {"train": [0] * args.epoch, "test": [0] * args.epoch}

    torch.backends.cudnn.benchmark = True

    if tfboard:
        if not os.path.isdir(tfboard):
            os.makedirs(tfboard)
        tblogger = SummaryWriter(tfboard)

    for epoch in range(args.epoch):
        print("Epoch:{}/{}".format(epoch + 1, args.epoch))
        print("-----------")

        for phase in phase_list:
            if (phase == "train") and (epoch == 0):
                continue
            else:
                epoch_loss = 0
                epoch_correct = 0

                if phase == "train":
                    net.train()
                    torch.set_grad_enabled(True)
                else:
                    net.eval()
                    torch.set_grad_enabled(False)

                for i, (img0, img1, labels) in enumerate(dataloaders_dict[phase]):
                    if phase == "train":
                        tblogger.add_images(f"{epoch}/crop_flip", img0[:8])
                        tblogger.add_images(f"{epoch}/colorjitter", img1[:8])
                    optimizer.zero_grad()
                    img01 = torch.cat([img0, img1], axis=0).to(device)
                    labels = labels.to(device)
                    out, middle = net(img01)
                    out0, out1 = torch.split(out, img0.size()[0], dim=0)

                    epoch_feature = (
                        middle if i == 0 else torch.cat([epoch_feature, middle], dim=0)
                    )
                    epoch_label = (
                        labels if i == 0 else torch.cat([epoch_label, labels], dim=0)
                    )

                    # if args.train_mode == 0:
                    loss = ContrastiveLoss(out0, out1)
                    if args.train_mode == 1:
                        out = net.fc3(middle)
                        out0, out1 = torch.split(out, img0.size()[0], dim=0)
                        for out_ in [out0, out1]:
                            loss += args.alpha * criterion(out_, labels)

                            _, preds = torch.max(out_, 1)
                            epoch_correct += torch.sum(preds == labels.data)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += float(loss.item()) * img0.size(0)

                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                if args.train_mode == 1:
                    epoch_correct = epoch_correct.double() / (
                        len(dataloaders_dict[phase].dataset) * 2
                    )

                Loss[phase][epoch] = epoch_loss
                if args.train_mode == 0:
                    print("{} Loss:{:.4f} ".format(phase, epoch_loss))
                else:
                    print(
                        "{} Loss:{:.4f} Acc:{:.4f} ".format(
                            phase, epoch_loss, epoch_correct
                        )
                    )

                if tfboard:
                    tblogger.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
                    tblogger.add_scalar(f"{phase}/Acc", epoch_correct, epoch)

                ind_rand = np.random.randint(
                    0, len(dataloaders_dict[phase].dataset), size=600
                )

                epoch_label = epoch_label.cpu().data.numpy()[ind_rand]
                # epoch_label = epoch_label.cpu().data.numpy()
                tsne = TSNE(n_components=2).fit_transform(
                    epoch_feature.cpu().data.numpy()[ind_rand]
                )
                emb_0 = tsne[np.where(epoch_label == 0)[0]]
                emb_1 = tsne[np.where(epoch_label == 1)[0]]
                emb_2 = tsne[np.where(epoch_label == 2)[0]]
                save_embedding(
                    args, emb_0, emb_1, emb_2, tblogger, epoch, phase, emb_save_dir
                )
                if epoch % 5 == 0:
                    torch.save(
                        net.state_dict(),
                        os.path.join(weight_save_dir, f"{epoch}_weight.pth"),
                    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    img_path = []
    label = []

    transforms = ImageTransform(batchsize=args.batchsize)

    classes = ["no", "yes", "garbage"]
    img_dirs = glob(os.path.join(args.input, "*"))
    for img_dir in img_dirs:
        paths = sorted(glob(os.path.join(img_dir, "*")))
        random.shuffle(paths)
        paths = paths[:1000]
        img_path.extend(paths)
        for i in range(len(paths)):
            label.append(classes.index(os.path.basename(img_dir)))

    print("##Label")
    print("|-- Yes: ", label.count(1))
    print("|-- No : ", label.count(0))
    print("|-- Garbage: ", label.count(2))

    data = dict()
    x_train, x_test, y_train, y_test = train_test_split(img_path, label, test_size=0.20)
    data["x_train"] = x_train
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    with open(os.path.join(args.output, "data.json"), "w") as f:
        json.dump(data, f, indent=4)

    net = ContrastiveResnetModel(out_dim=args.n_cls)
    net.to(device)

    for name, param in net.named_parameters():
        param.require_grad = True

    train_dataset = ImageDataset(x_train, y_train, transform=transforms, phase='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, num_workers=1, shuffle=True
    )

    test_dataset = ImageDataset(x_test, y_test, transform=transforms, phase='train')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )
    print("## Dataset")
    print("|-- Train_Length: ", len(train_dataloader.dataset))
    print("|-- Test_Length: ", len(test_dataloader.dataset))

    dataloaders_dict = {"train": train_dataloader, "test": test_dataloader}

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    trainer(
        args,
        net,
        dataloaders_dict,
        output=args.output,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        tfboard=(args.output + "/tfboard"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", type=int, help="0:con, 1:joint")
    parser.add_argument(
        "--input", type=str, default="/mnt/aoni02/matsunaga/200313_global-model/train"
    )
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tfboard", type=strtobool)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_cls", type=int)
    parser.add_argument("--gpuid", type=str)
    parser.add_argument("--alpha", type=int, default=100)

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main(args)
