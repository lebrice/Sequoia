from setuptools import setup, find_packages
import os
import versioneer

with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r") as file:
    lines = [ln.strip() for ln in file.readlines()]

packages_to_export = find_packages(
    where=".", exclude=["tests*", "examples*"], include="sequoia*"
)

required_packages = [
    line for line in lines if line and not line.startswith("#")
]

extras_require = {
    "monsterkong": [
        "meta_monsterkong @ git+https://github.com/lebrice/MetaMonsterkong.git#egg=meta_monsterkong"
    ],
    "atari": [
        "gym[atari] @ git+https://www.github.com/lebrice/gym@easier_custom_spaces#egg=gym"
    ],
    "hpo": ["orion", "orion.algo.skopt"],
    "avalanche": [
        "gdown",  # BUG: Avalanche needs this to download cub200 dataset.
        "avalanche @ git+https://github.com/ContinualAI/avalanche.git#egg=avalanche",
    ],
    # NOTE: Removing this for now, because it has very strict requirements, and includes
    # a lot of copy-pasted code, and doesn't really add anything compared to metaworld.
    # This isn't right.
    # "mtenv": [
    #     "mtenv @ git+https://github.com/facebookresearch/mtenv.git@main#egg='mtenv[metaworld]'"
    # ],
    "metaworld": [
        "metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld"
    ],
}
# Add-up all the optional requirements, and then remove any duplicates.
extras_require["all"] = list(
    set(
        sum(
            [
                [extra_requirements]
                if isinstance(extra_requirements, str)
                else extra_requirements
                for extra_requirements in extras_require.values()
            ],
            [],
        )
    )
)

setup(
    name="sequoia",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="The Research Tree - A playground for research at the intersection of Continual, Reinforcement, and Self-Supervised Learning.",
    url="https://github.com/lebrice/Sequoia",
    author="Fabrice Normandin",
    author_email="fabrice.normandin@gmail.com",
    license="GPLv3",
    packages=packages_to_export,
    extras_require=extras_require,
    install_requires=required_packages,
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
            "sequoia_sweep = sequoia.experiments.hpo_sweep:main",
        ],
    },
)
