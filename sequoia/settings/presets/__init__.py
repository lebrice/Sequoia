import os
from pathlib import Path

presets_dir = Path(os.path.dirname(__file__))

setting_presets = {
    "cartpole_state": presets_dir / "cartpole_state.yaml",
    "cartpole_pixels": presets_dir / "cartpole_pixels.yaml",
    "mnist": presets_dir / "mnist.yaml",
    "fashion_mnist": presets_dir / "fashion_mnist.yaml",
    "cifar10": presets_dir / "cifar10.yaml",
    "cifar100": presets_dir / "cifar100.yaml",
    "monsterkong": presets_dir / "monsterkong_pixels.yaml",
}
