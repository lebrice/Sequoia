import os
from pathlib import Path

presets_dir = Path(os.path.dirname(__file__))

setting_presets = {
    file.stem: file    
    for file in presets_dir.glob("*.yaml")
}