from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r") as file:
    lines = [ln.strip() for ln in file.readlines()]

packages_pip = []

for ln in lines:
    if ln.startswith("#"):
        continue
    if ln:  # if requirement is not empty
        packages_pip.append(ln)

packages_git = []

setup(
    name="sequoia",
    version="0.0.1",
    description="The Research Tree - A playground for research at the intersection of Continual, Reinforcement, and Self-Supervised Learning.",
    url="https://github.com/lebrice/Sequoia",
    author="Fabrice Normandin",
    author_email="fabrice.normandin@gmail.com",
    license="GPLv3",
    packages=[package for package in find_packages() if package.startswith("sequoia")],
    extras_require={
        "monsterkong": [
            "meta_monsterkong @ git+https://github.com/lebrice/MetaMonsterkong.git#egg=meta_monsterkong"
        ],
        "atari": [
            "gym[atari] @ git+https://www.github.com/lebrice/gym@easier_custom_spaces#egg=gym"
        ],
        "hpo": ["orion", "orion.algo.skopt"],
        "mtenv": [
            "mtenv @ git+https://github.com/facebookresearch/mtenv.git@main#egg='mtenv[metaworld]'"
        ],
        "metaworld": [
            "metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld"
        ],
    },
    install_requires=packages_pip,
    dependency_links=packages_git,
    python_requires=">=3.7",
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points={
        "console_scripts": [
            "sequoia = sequoia.experiments.experiment:main",
            # TODO: This entry-point is added temporarily while we redesign the
            # command-line API (See https://github.com/lebrice/Sequoia/issues/47)
            "sequoia_sweep = sequoia.experiments.hpo_sweep:main"
        ],
    }
)
