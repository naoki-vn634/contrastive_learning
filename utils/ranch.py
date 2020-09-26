import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    default="/mnt/aoni02/matsunaga/Dataset/200313_global-model-ranch/garbage_ver3/train/no",
)
parser.add_argument(
    "--output",
    type=str,
    default="/mnt/aoni02/matsunaga/Dataset/200313_global-model-ranch",
)
parser.add_argument("--phase", type=str)
args = parser.parse_args()

num_list = list()
ranch = glob(os.path.join(args.input, "*"))
for r in ranch:
    img = glob(os.path.join(r, "*"))
    num_list.append(len(img))

all = str(np.sum(np.array(num_list)))


ranch_name = [os.path.basename(r) for r in ranch]
plt.figure(figsize=(20, 10))
plt.title(f"{args.phase}_ranch(num:{all})")
plt.bar(ranch_name, np.array(num_list))
plt.savefig(os.path.join(args.output, f"{args.phase}_count_ranch.png"))
