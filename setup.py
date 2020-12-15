from setuptools import setup, find_packages
from sequoia import __version__
import os

# TODO: Figure out how to specify 'extras'
extras = {
#     "ewc": "git+https://github.com/oleksost/nngeometry.git"
}

PATH_ROOT = os.path.dirname(__file__)

def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    """ Taken from the pl_bolts repo.
    """ 
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[:ln.index(comment_char)].strip()
        if ln.startswith('http'):  # skip directly installed dependencies
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name='sequoia',
    version=__version__,
    description="The Research Tree - A playground for research at the intersection of Continual, Reinforcement, and Self-Supervised Learning.",
    url='https://github.com/lebrice/Sequoia',
    author='Fabrice Normandin',
    author_email='fabrice.normandin@gmail.com',
    license='',
    packages=[package for package in find_packages()
                if package.startswith('sequoia')],
    extras_require=extras,
    install_requires=load_requirements(),
    python_requires='>=3.7',
    tests_require=['pytest'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)