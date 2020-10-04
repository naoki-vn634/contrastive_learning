import argparse
import os
import pickle
import shutil
import torch
import torch.nn as nn
from distutils.util import strtobool
from math import log2

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def thu_cost_relation(cost_rec_ent, cost_pre_ent, thu_rec_ent, thu_pre_ent):
    plt.figure()
    plt.plot(thu_rec_ent[1], cost_rec_ent[1], label="recall", alpha=1.0, color="blue")
    plt.plot(
        thu_pre_ent[1], cost_pre_ent[1], label="precision", alpha=1.0, color="green"
    )
    plt.grid(which="major", color="gray", linestyle="-")
    # plt.legend()
    plt.xlabel("thureshold")
    plt.ylabel("cost")
    plt.title("Cost - thureshold relation")
    plt.savefig(os.path.join(args.output, "cost_thureshold_relation_proposed.png"))
    plt.close()


def calculate_entropy(pos):
    entropy = np.array([])

    for posterior in pos:
        ent = -sum([x * log2(x) if x != 0 else 0 for x in posterior])
        entropy = np.append(entropy, ent)
    return entropy


def rec_result(cost_rec_ent, rec_ent, thu_rec_ent, output, compare, shuffles):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    colors = ["blue", "red", "black"]
    for i, (cost, recall) in enumerate(zip(cost_rec_ent, rec_ent)):
        ax1.plot(cost, recall, label=compare[i], alpha=0.7, color=colors[i])
    if args.random_sampling:
        ax1.plot(shuffles[0], shuffles[2], label="random", alpha=0.7, color="green")
    ax2.plot(
        cost_rec_ent[0],
        thu_rec_ent[0],
        label="baseline",
        linestyle="dashed",
        color="blue",
        alpha=0.7,
    )
    ax2.plot(
        cost_rec_ent[1],
        thu_rec_ent[1],
        label="proposed",
        linestyle="dashed",
        color="red",
        alpha=0.7,
    )
    ax1.grid(which="major", color="gray", linestyle="-", alpha=0.5)
    plt.xlabel("Cost")
    ax1.legend(loc="lower center")
    ax1.set_ylabel("recall")
    ax2.set_ylabel("thureshold")
    plt.title("Recall Thureshold")
    plt.savefig(os.path.join(output, "recall_thureshold.png"))
    plt.close()


def pre_result(cost_pre_ent, pre_ent, thu_pre_ent, output, compare, shuffles):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    colors = ["blue", "red", "black"]
    for i, (cost, precision) in enumerate(zip(cost_pre_ent, pre_ent)):
        ax1.plot(cost, precision, label=compare[i], alpha=0.7, color=colors[i])
    if args.random_sampling:
        ax1.plot(shuffles[0], shuffles[1], label="random", alpha=0.7, color="green")
    ax2.plot(
        cost_pre_ent[0],
        thu_pre_ent[0],
        label="baseline",
        color="blue",
        linestyle="dashed",
        alpha=0.7,
    )
    ax2.plot(
        cost_pre_ent[1],
        thu_pre_ent[1],
        label="proposed",
        color="red",
        linestyle="dashed",
        alpha=0.7,
    )
    ax1.grid(which="major", color="gray", linestyle="-", alpha=0.5)
    plt.xlabel("Cost")
    ax1.legend(loc="lower center")
    ax1.set_ylabel("precision")
    ax2.set_ylabel("thureshold")
    plt.title("precision Thureshold")
    plt.savefig(os.path.join(output, "precision_thureshold.png"))
    plt.close()


def load_data(dir):
    pos = np.load(os.path.join(dir, "posterior.npy"))
    pred = np.load(os.path.join(dir, "y_pred.npy"))
    true = np.load(os.path.join(dir, "y_true.npy"))
    ent = calculate_entropy(pos)

    return pos, pred, true, ent


def convert_pos(posterior):
    for i, pos in enumerate(posterior):
        pos_conv = np.expand_dims(np.array([1 - pos[1], pos[1]]), axis=0)
        pos_c = np.concatenate([pos_c, pos_conv], axis=0) if i != 0 else pos_conv

    return pos_c


def caliblate_t(logits, pred, temperature):
    for i, t in enumerate(temperature):
        logits = logits / t
        pos = nn.functional.softmax(torch.from_numpy(logits), dim=1)
        pos_c = convert_pos(pos.cpu().data.numpy())
        ent = np.expand_dims(calculate_entropy(pos_c), axis=0)
        poses = pos_c if i == 0 else np.concatenate([poses, pos_c], axis=0)
        entropies = ent if i == 0 else np.concatenate([entropies, ent], axis=0)

    return poses, entropies


def calculate_cost(label, pred, entropy):
    rec_ = []
    pre_ = []
    cost_ = []
    thu_ = []

    for ent in entropy:
        rec = []
        pre = []
        cost = []
        thureshold = []
        max = np.nanmax(ent)
        min = np.nanmin(ent)

        for thu in np.arange(max, min - 0.02, -0.01):
            query = np.where(ent >= thu)[0]
            cost.append(len(query))
            rec_child = len(np.where(label[np.where(pred == label)[0]] == 1)[0])
            rec_mother = len(np.where(label == 1)[0])
            pre_child = len(np.where(pred[np.where(pred == label)[0]] == 1)[0])
            pre_mother = len(np.where(pred == 1)[0])
            for j in query:
                if pred[j] != label[j]:
                    if pred[j] == 0:
                        rec_child += 1
                    elif pred[j] == 1:
                        pre_child += 1
            pre.append(float(pre_child / pre_mother))
            rec.append(float(rec_child / rec_mother))
            thureshold.append(thu)
        thu_.append(np.array(thureshold))
        cost_.append(np.array(cost))
        rec_.append(np.array(rec))
        pre_.append(np.array(pre))
    return thu_, cost_, rec_, pre_


def temperature_ralation(thu, cost, recall, precision, temperature):
    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for c, r, p, t in zip(cost, recall, precision, temperature):
        ax1.plot(c, r, label=str(t), alpha=0.7)
        ax2.plot(c, p, label=str(t), alpha=0.7)

    ax1.set_xlabel("cost")
    ax2.set_xlabel("cost")
    ax1.set_ylabel("recall")
    ax2.set_ylabel("precision")
    ax1.legend()
    ax2.legend()
    ax1.grid(which="major", color="gray", linestyle="-", alpha=0.5)
    ax2.grid(which="major", color="gray", linestyle="-", alpha=0.5)
    ax1.set_title("Recall")
    ax2.set_title("Precision")
    plt.savefig("./temperature_ralation.png")


def main(args):
    classes = ["no_id", "yes_id"]
    val_path = glob("/mnt/aoni02/matsunaga/10_cropped-images/all_id/*/*.jpg")
    val_label = [
        classes.index(os.path.basename(os.path.dirname(path))) for path in val_path
    ]
    val_score = np.load(args.score)
    val_preds = np.load(args.pred)

    thu, cost, recall, precision = calculate_cost(val_label, val_preds, val_score)

    temperature_ralation(thu, cost, recall, precision, temperature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparison method")
    parser.add_argument("--score", type=str)
    parser.add_argument("--", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    main(args)
