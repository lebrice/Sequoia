import argparse
from datasets import Mnist, FashionMnist, Cifar10, Cifar100
from pathlib import Path
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=Path, default="data", help="path to download the datasets to.")
args = parser.parse_args()
data_dir: Path = args.data_dir
print("Downloading datasets to data dir: ", data_dir)
for dataset_class in [Mnist, FashionMnist, Cifar10, Cifar100]:
    dataset = dataset_class()
    dataset.load(data_dir=data_dir)
