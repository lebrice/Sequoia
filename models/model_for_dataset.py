from datasets.datasets import Datasets
from .classifier import Classifier
from .cifar import Cifar10Classifier, Cifar100Classifier
from .mnist import MnistClassifier
from typing import Dict, Type, Union

model_class_for_dataset: Dict[Datasets, Type[Classifier]] = {
    Datasets.mnist: MnistClassifier,
    Datasets.fashion_mnist: MnistClassifier,
    Datasets.cifar10: Cifar10Classifier,
}

def get_model_class_for_dataset(dataset: Union[str, Datasets]) -> Type[Classifier]:
    if isinstance(dataset, str):
        dataset = Datasets[dataset.lower().replace("-", "_")]
    if dataset not in model_class_for_dataset:
        raise RuntimeError(f"No model available for dataset {dataset}")
    return model_class_for_dataset[dataset]