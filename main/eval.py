import argparse
import os
import sys
import torch
import numpy as np


from glob import glob
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append("../models/")
from model import ContrastiveResnetModel

sys.path.append("../preprocess/")
from img_preprocess import ImageDataset, ImageTransform


def main(args):
    y_true, y_pred = np.array([]), np.array([])
    label = []

    net = ContrastiveResnetModel(num_channels=args.n_cls, hidden=args.hidden)
    net.load_state_dict(torch.load(args.weight))

    val_path = glob(
        os.path.join("/mnt/aoni02/matsunaga/10_cropped-images/all_id/*/*.jpg")
    )
    for path in val_path:
        label.append(1) if "yes" in path else label.append(0)

    transforms = ImageTransform(batchsize=args.batchsize)
    val_dataset = ImageDataset(val_path, label, transform=transforms, phase="val")
    val_dataloader = torch.utils.data.Dataloader(
        val_dataset, batch_size=args.batchsize, num_workers=40, shuffle=False
    )

    for image, label in val_dataloader:
        out, middle = net(image)
        if args.hidden == 0:
            cls_out = net.fc3(middle)
        elif args.hidden == 1:
            cls_out = net.fc4(out)

        _, preds = torch.max(cls_out, 1)
        y_pred = np.append(y_pred, preds.cpu().data.numpy())
        y_true = np.append(y_true, label.cpu().data.numpy())

    confmat = confusion_matrix(y_true, y_pred)
    print(confmat)
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--hidden", type=int, help="0:2048, 1:128")
    parser.add_argument("--weight", type=str)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--n_cls", type=int)

    args = parser.parse_args()
    main(args)