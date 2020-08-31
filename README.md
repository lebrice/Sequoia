# SSCL
Self-Supervised Learning for Continual Learning.



## Installation
Requires python >= 3.7

```console
git submodule init
git submodule update
pip install -r requirements.txt
```

## TODOs:
- [ ] Write some tests for **every single module**. Have them be easy to read so people could ideally understand how things work by simply reading the tests.
- [ ] Create a `methods` folder to house standalone methods that can target different settings and add some examples there.
- [ ] Fix the checkpoint directory used by pytorch-lightning

## Getting Started:
- Take a look at Pytorch Lightning
- Take a quick look at [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I've created) which we use to generate the command-line arguments for the experiments.
