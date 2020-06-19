from datasets import Datasets, DatasetConfig
from .classifier import Classifier
from .cifar import Cifar10Classifier, Cifar100Classifier
from .mnist import MnistClassifier
from .imagenet import ImageNetClassifier
from typing import Dict, Type, Union

model_class_for_dataset: Dict[Datasets, Type[Classifier]] = {
    Datasets.mnist: MnistClassifier,
    Datasets.fashion_mnist: MnistClassifier,
    Datasets.cifar10: Cifar10Classifier,
    Datasets.imagenet: ImageNetClassifier,
}

def get_model_class_for_dataset(dataset: Union[str, Datasets, DatasetConfig]) -> Type[Classifier]:
    try:
        if isinstance(dataset, str):
            dataset = Datasets[dataset.lower().replace("-", "_")]
        if isinstance(dataset, DatasetConfig):
            k = [v.name for v in Datasets if v.value == dataset][0]
            print(k)
            dataset = Datasets[k]
        return model_class_for_dataset[dataset]
    except KeyError as e:
        raise RuntimeError(f"No model available for dataset {dataset}.")
