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
- **Rewrite this outdated README**
- Write some tests for LossInfo in `common/losses_test.py` to make it easier to
    understand for people coming in.
- Add some kind of /methods and /settings folders.


## Debugging / Running experiments:
run `python main.py <setting_name> [--<hparam_name> <hparam_value>]`.

Different experimental settings are located the `experiments` folder.
There are currently three settings to choose from:
- "iid":
    - The usual iid setting. Nothing fancy.
- "task-incremental":
    - Learning occurs as a series of tasks, each with a subset of the classes.
    For example, first only 0's and 1's, then 2's and 3's, etc.

    - For each task, if at least one auxiliary tasks is enabled (see below), we first do `--unsupervised_epochs` epochs of self-supervised training, followed by `--supervised_epochs` epochs of supervised training.

    - (Could be renamed to class-incremental, since the same image is always the same label)
- "class-incremental"
    - (deprecated)
    - (Should be renamed to something like "online-class-incremental")

To learn more about the command-line options available, use `"python <setting-name> --help"`, or check out the different classes in the [experiments folder](/experiments).


## Adding auxiliary tasks:
By default, the model uses no auxiliary tasks, and is just a simple classifier. It therefore can only do supervised training.

To add auxiliary tasks, use the `"--<task_name>.coefficient <value>"` argument.
This value is used to scale the task loss before it is added to the total loss and backpropagted. The default value for all tasks is 0, meaning they are disabled by default.

## Examples:
- Running the baseline in a setting:
    ```console
    python main.py <setting>
    ```
- Local debugging run (no wandb):
    ```console
    python main.py <setting> --debug
    ```
- Adding a reconstruction auxiliary task with a VAE:
    ```console
    python main.py <setting> --reconstruction.coefficient 1
    ```
- Adding Rotation
    ```console
    python main.py <setting> --rotation.coefficient 1
    ```
- Training the feature extractor without the classifier gradients (only through self-supervision):
    ```console
    python main.py <setting> [--<aux_task>.coefficient <value>, ...] --detach_classifier
    ```


## TODOS:
- Debug the VAE, figure out why the samples don't look good even in iid setting.
- Debug the task-incremental setting (dataset setup, etc.)
- Add plots:
  - OML-style plot showing evolution of training and validation error during class-incremental setup.
  - Plot showing the evolution of the training/validation accuracy over the course of class-incremental training for both the baseline and self-supervised models 
  - Plots showing the loss of an auxiliary task in the i.i.d. setting versus non-iid:
    - Might be useful for arguing that the data can be considered IID from the perspective of the aux tasks
    - Have to be careful about the relationship between task and dataset (ex: the "0" class in MNIST and Rotation task)


## Running Experiments


The main executable is `main.py`. 

There are currently three settings:
- iid:

- i.i.d setting (baseline)
    ```console
    python main.py iid
    ```

- i.i.d setting (self-supervised)
    ```console
    python main.py iid --reconstruction
    ```

- i.i.d setting (self supervised model)
    ```console
    python main.py baseline_aux
    ```

- Class-incremental setting (baseline) 
    ```console
    python main.py baseline --class-incremental
    ```

- Class-incremental setting () 
    ```console
    python main.py class-incremental
    ```

Note that we use [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I'm currently developing) to generate the command-line arguments for the experiments.
