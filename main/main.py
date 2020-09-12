import argparse
import cv2
import json
import os
import random
import sys
from distutils.util import strtobool
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


def trainer(
    args,
    net,
    dataloaders_dict,
    output,
    optimizer,
    scheduler,
    criterion,
    device,
    tfboard,
):
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
        # scheduler.step()

        for phase in phase_list:
            bar = tqdm(total=len(dataloaders_dict[phase].dataset))
            if (phase == "train") and (epoch == 0):
                continue
            if (phase == "test") and (epoch % 5 != 0):
                continue

            print(f"Phase:{phase}")
            epoch_loss, epoch_correct, epoch_con, epoch_cls = 0, 0, 0, 0

            if phase == "train":
                net.train()
                torch.set_grad_enabled(True)
            else:
                net.eval()
                torch.set_grad_enabled(False)

            for i, (img0, img1, labels) in enumerate(dataloaders_dict[phase]):

                bar.update(img0.size(0))
                if phase == "train":
                    tblogger.add_images(f"{epoch}/crop_flip", img0[:8])
                    tblogger.add_images(f"{epoch}/colorjitter", img1[:8])
                optimizer.zero_grad()
                img01 = torch.cat([img0, img1], axis=0).to(device)
                labels = labels.to(device)

                out, middle = net(img01)
                out0, out1 = torch.split(out, img0.size()[0], dim=0)
                L_con, L_cls = 0, 0

                if args.train_mode != 2:
                    L_con = ContrastiveLoss(out0, out1)
                if args.train_mode != 0:
                    if args.hidden == 0:
                        out = net.fc3(middle)
                    elif args.hidden == 1:
                        out = net.fc4(out)
                    out0, out1 = torch.split(out, img0.size()[0], dim=0)

                    for out_ in [out0, out1]:
                        L_cls += args.alpha * criterion(out_, labels)
                        _, preds = torch.max(out_, 1)
                        epoch_correct += torch.sum(preds == labels.data)

                loss = L_con + L_cls

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                epoch_loss += float(loss.item()) * img0.size(0)
                if args.train_mode != 2:
                    epoch_con += float(L_con.item()) * img0.size(0)
                if args.train_mode != 0:
                    epoch_cls += float(L_cls.item()) * img0.size(0)
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            if args.train_mode != 2:
                epoch_con = epoch_con / len(dataloaders_dict[phase].dataset)
            if args.train_mode != 0:
                epoch_cls = epoch_cls / len(dataloaders_dict[phase].dataset)

            if args.train_mode != 0:
                epoch_correct = epoch_correct.double() / (
                    len(dataloaders_dict[phase].dataset) * 2
                )

            Loss[phase][epoch] = epoch_loss
            if args.train_mode == 0:
                print("Loss: {:.4f}".format(epoch_loss))
                print("|-L_con: {:.4f}".format(epoch_con))
                print("|-L_cls: {:.4f}".format(epoch_cls))
            else:
                print("Acc:{:.4f} ".format(epoch_correct))
                print("Loss: {:.4f}".format(epoch_loss))
                if args.train_mode != 2:
                    print("|-L_con: {:.4f}".format(epoch_con))
                if args.train_mode != 0:
                    print("|-L_cls: {:.4f}".format(epoch_cls))

            if tfboard:
                tblogger.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
                if args.train_mode == 1:
                    tblogger.add_scalar(f"{phase}/Acc", epoch_correct, epoch)

            # ind_rand = np.random.randint(
            #     0, len(dataloaders_dict[phase].dataset), size=600
            # )

            # epoch_label = epoch_label.cpu().data.numpy()[ind_rand]
            # epoch_label = epoch_label.cpu().data.numpy()
            # tsne = TSNE(n_components=2).fit_transform(
            #     epoch_feature.cpu().data.numpy()[ind_rand]
            # )
            # emb_0 = tsne[np.where(epoch_label == 0)[0]]
            # emb_1 = tsne[np.where(epoch_label == 1)[0]]
            # emb_2 = tsne[np.where(epoch_label == 2)[0]]
            # save_embedding(
            #     args, emb_0, emb_1, emb_2, tblogger, epoch, phase, emb_save_dir
            # )
            if epoch % args.checkpoint_iter == 0:
                torch.save(
                    net.state_dict(),
                    os.path.join(weight_save_dir, f"{epoch}_weight.pth"),
                )
            save_epoch = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 90]
            if epoch in save_epoch:
                torch.save(
                    net.state_dict(),
                    os.path.join(weight_save_dir, f"{epoch}_weight.pth"),
                )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)

    if args.train_mode == 1:
        with open(os.path.join(args.input, "data.json")) as f:
            data = json.load(f)
            x_train, x_test = data["x_train"], data["x_test"]
            y_train, y_test = data["y_train"], data["y_test"]

    else:  # pretrain / only classification
        img_path = []
        label = []

        classes = ["no", "yes"]
        img_dirs = glob(os.path.join(args.input, "*"))
        for img_dir in img_dirs:
            if os.path.basename(img_dir) == "garbage":
                continue
            paths = sorted(glob(os.path.join(img_dir, "*")))
            random.shuffle(paths)
            ext_path = paths[:5000]
            img_path.extend(ext_path)
            for i in range(len(ext_path)):
                label.append(classes.index(os.path.basename(img_dir)))

        data = dict()
        x_train, x_test, y_train, y_test = train_test_split(
            img_path, label, test_size=0.20
        )
        data["x_train"] = x_train
        data["x_test"] = x_test
        data["y_train"] = y_train
        data["y_test"] = y_test
        with open(os.path.join(args.output, "data.json"), "w") as f:
            json.dump(data, f, indent=4)

    # Define Model
    net = ContrastiveResnetModel(out_dim=args.n_cls, hidden=args.hidden)
    net.to(device)
    if args.train_mode == 1:
        net.load_state_dict(torch.load(args.weight))

    transforms = ImageTransform(batchsize=args.batchsize)
    for name, param in net.named_parameters():
        param.require_grad = True

    train_dataset = ImageDataset(x_train, y_train, transform=transforms, phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, num_workers=40, shuffle=True
    )

    test_dataset = ImageDataset(x_test, y_test, transform=transforms, phase="train")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, num_workers=40, shuffle=False
    )
    print("## Dataset")
    print("|-- Train_Length: ", len(train_dataloader.dataset))
    print("|-- Test_Length: ", len(test_dataloader.dataset))

    dataloaders_dict = {"train": train_dataloader, "test": test_dataloader}

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6
        )

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=20, eta_min=0.0001
    # )
    scheduler = None
    criterion = nn.CrossEntropyLoss()

    trainer(
        args,
        net,
        dataloaders_dict,
        output=args.output,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        tfboard=(args.output + "/tfboard"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_mode", type=int, help="0:contrastive, 1:joint, 2:class_only"
    )
    parser.add_argument(
        "--input", type=str, default="/mnt/aoni02/matsunaga/200313_global-model/train"
    )
    parser.add_argument("--hidden", type=int, help="0:2048, 1:128")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tfboard", type=strtobool)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_cls", type=int)
    parser.add_argument("--gpuid", type=str)
    parser.add_argument("--checkpoint_iter", type=int, default=10)
    parser.add_argument("--alpha", type=int, default=100)
    parser.add_argument("--optim", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main(args)
