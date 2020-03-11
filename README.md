# SSCL
Self-Supervised Continual Learning


## Debugging / Running experiments:
run `python main.py <setting_name> [--<hparam_name> <hparam_value>]`.

Different experimental settings are located the `experiments` folder.
There are currently three settings to choose from:
- "iid":
  - Your usual iid setting.
- "class_incremental"
  - (Should be renamed to something like "online_class_incremental")
  - Learning occurs as a stream of first only 0's and 1's, then 2's and 3's, etc.
  - Every datapoint is only seen once.
- "task_incremental" (WIP)
  - (Could be renamed to class_incremental)
  - Same as above, but can perform as many epochs on a given "task" as you want.

By default, uses the baseline options (no auxiliary tasks, just a simple classifier).

To add auxiliary tasks, use `"--<task_name>.coefficient <value>"`. This value is used to scale the task loss before it is added to the total loss and backpropagted. The default value for all tasks is 0, meaning they are disabled by default.

Examples:
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
- Debug the task_incremental setting (dataset setup, etc.)
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
    python main.py baseline --class_incremental
    ```

- Class-incremental setting () 
    ```console
    python main.py class_incremental
    ```

Note that we use [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I'm currently developing) to generate the command-line arguments for the experiments.

## Installation
Requires python >= 3.6
- Python 3.6:
    ```console
    pip install dataclasses
    pip install -r requirements.txt
    ```
- Python >= 3.7:
    ```console
    pip install -r requirements.txt
    ```

