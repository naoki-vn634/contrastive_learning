import argparse
import os
import sys
import torch
import torch.nn as nn
from distutils.util import strtobool
from glob import glob
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

sys.path.append("../preprocess/")
from img_preprocess import ImageDataset, ImageTransform

sys.path.append("../models/")
from model import ContrastiveResnetModel


def trainer(args, net, dataloaders_dict, output, optimizer, criterion, device, tfboard):
    phase_list = ["train", "test"]
    Loss = {"train": [0] * args.epoch, "test": [0] * args.epoch}
    Acc = {"train": [0] * args.epoch, "test": [0] * args.epoch}

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

                for img0, img1, labels in dataloaders_dict[phase]:
                    optimizer.zero_grad()
                    img0 = img0.to(device)
                    img1 = img1.to(device)
                    labels = labels.to(device)
                    out, middle = net(images)
                    loss = criterion(out, labels)

                    _, preds = torch.max(out, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += float(loss.item()) * images.size(0)
                    epoch_correct += torch.sum(preds == labels.data)

                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_correct.double() / len(
                    dataloaders_dict[phase].dataset
                )

                Loss[phase][epoch] = epoch_loss
                Acc[phase][epoch] = epoch_acc
                print("{} Loss:{:.4f} Acc:{:.4f}".format(phase, epoch_loss, epoch_acc))

                # if tfboard:
                #     tblogger.add_scaler('{}/Loss'.format(phase),epoch_loss,epoch)
                #     tblogger.add_scaler('{}/Acc'.format(phase),epoch_acc,epoch)


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
        img_path.extend(paths)
        for i in range(len(paths)):
            label.append(classes.index(os.path.basename(img_dir)))

    print("##Label")
    print("|-- Yes: ", label.count(1))
    print("|-- No : ", label.count(0))
    print("|-- Garbage: ", label.count(2))

    x_train, x_test, y_train, y_test = train_test_split(img_path, label, test_size=0.20)
    net = ContrastiveResnetModel(num_classes=args.n_cls)
    net.to(device)

    for name, param in net.named_parameters():
        if "fc" in name:
            param.require_grad = True
        else:
            param.require_grad = False

    train_dataset = ImageDataset(x_train, y_train, transform=transforms)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, num_workers=1, shuffle=True
    )

    test_dataset = ImageDataset(x_test, y_test, transform=transforms)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )
    print("## Dataset")
    print("|-- Train_Length: ", len(train_dataloader.dataset))
    print("|-- Test_Length: ", len(test_dataloader.dataset))

    dataloaders_dict = {"train": train_dataloader, "test": test_dataloader}

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, momentum=0.9)
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
    parser.add_argument(
        "--input", type=str, default="/mnt/aoni02/matsunaga/200313_global-model/train"
    )
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tfboard", type=strtobool)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n_cls", type=int)
    parser.add_argument("--gpuid", type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main(args)
