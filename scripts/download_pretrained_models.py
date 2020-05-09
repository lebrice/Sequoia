import torchvision.models as models
from pathlib import Path
import os


def download_pretrained_models(save_dir: Path=None) -> None:
    """Downloads all the pretrained models that might be needed later.

    TODO: Save them to `save_dir` instead of the default location (which is
    $TORCH_HOME/checkpoints, afaik.)
    """
    if save_dir:
        os.environ["TORCH_HOME"] = str(save_dir)

    save_dir_str = os.environ.get("TORCH_HOME")
    print(f"Save dir: '{save_dir_str}'")
    
    all_models = [
        models.vgg16,  # This is the only one tested so far.
        models.resnet18,
        models.resnet34,
        models.resnet50,
        models.resnet101,
        models.resnet152,
        models.alexnet,
        models.squeezenet1_0,  # Not supported yet (weird output shape)
        models.densenet161,
        models.inception_v3,  # Not supported yet (creating model takes forever?)
        models.googlenet,  # Not supported yet (creating model takes forever?)
        models.shufflenet_v2_x1_0,
        models.mobilenet_v2,
        models.resnext50_32x4d,
        models.wide_resnet50_2,
        models.mnasnet1_0,
    ]
    for model in all_models:
        print(f"Downloading model {model.__name__}")
        _ = model(pretrained=True, progress=True)

if __name__ == "__main__":
    # from argparse import ArgumentParser
    # parser = ArgumentParser(description=__doc__)
    # parser.add_argument("--save_dir", type=Path, default=None, help="Path to save the trained weights.")
    # args = parser.parse_args()
    # download_pretrained_models(save_dir=args.save_dir)
    download_pretrained_models()
