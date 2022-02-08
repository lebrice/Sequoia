import os
from pathlib import Path
from typing import Dict

presets_dir = Path(os.path.dirname(__file__))

setting_presets: Dict[str, Path] = {file.stem: file for file in presets_dir.rglob("*.yaml")}
