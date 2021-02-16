from setuptools import setup, find_packages

import os


PATH_ROOT = os.path.dirname(__file__)

def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    """ Taken from the pl_bolts repo.
    """ 
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if ln.startswith(comment_char):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

setup(
    name='sequoia',
    __version__ = "0.0.1",
    description="The Research Tree - A playground for research at the intersection of Continual, Reinforcement, and Self-Supervised Learning.",
    url='https://github.com/lebrice/Sequoia',
    author='Fabrice Normandin',
    author_email='fabrice.normandin@gmail.com',
    license='GPLv3',
    packages=[package for package in find_packages()
                if package.startswith('sequoia')],
    extras_require={
        "rl": [
            "meta_monsterkong @ git+https://github.com/mattriemer/monsterkong_examples.git@sequoia_integration#egg=meta_monsterkong"
        ],
        "hpo": [
            "orion",
            "orion.algo.skopt",
        ]},
    install_requires=load_requirements(),
    python_requires='>=3.7',
    tests_require=['pytest'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
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