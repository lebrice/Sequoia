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


setup(
    name="sequoia",
    version="0.0.2",
    description="The Research Tree - A playground for research at the intersection of Continual, Reinforcement, and Self-Supervised Learning.",
    url="https://github.com/lebrice/Sequoia",
    author="Fabrice Normandin",
    author_email="fabrice.normandin@gmail.com",
    license="GPLv3",
    packages=[package for package in find_packages() if package.startswith("sequoia")],
    extras_require={
        "rl": [
            "meta_monsterkong @ git+https://github.com/mattriemer/monsterkong_examples.git@sequoia_integration#egg=meta_monsterkong"
        ],
        "hpo": ["orion", "orion.algo.skopt",],
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
    entry_points={"console_scripts": ["sequoia = sequoia.main:main",],},
)
