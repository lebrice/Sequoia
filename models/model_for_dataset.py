from typing import Dict, Type, Union

from datasets import DatasetConfig
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)

from .cifar import Cifar10Classifier, Cifar100Classifier
from .classifier import Classifier
from .imagenet import ImageNetClassifier
from .mnist import MnistClassifier

model_class_for_dataset: Dict[LightningDataModule, Type[Classifier]] = {
    MNISTDataModule: MnistClassifier,
    FashionMNISTDataModule: MnistClassifier,
    CIFAR10DataModule: Cifar10Classifier,
    # Datasets.cifar100: Cifar100Classifier,
    ImagenetDataModule: ImageNetClassifier,
}

def get_model_class_for_dataset(dataset: DatasetConfig) -> Type[Classifier]:
    assert isinstance(dataset, DatasetConfig)
    print(dataset)
    try:

        k = [v.name for v in Datasets if v.value == dataset][0]
        print(k)
        dataset = Datasets[k]
        return model_class_for_dataset[dataset]
    except KeyError as e:
        raise RuntimeError(f"No model available for dataset {dataset}.")
