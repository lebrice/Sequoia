# SSCL
Self-Supervised Continual Learning


## TODOS:
- Add plots:
  - Plot showing the evolution of the training/validation accuracy over the course of class-incremental training for both the baseline and self-supervised models 
  - Plots showing the loss of an auxiliary task in the i.i.d. setting versus non-iid:
    - Might be useful for arguing that the data can be considered IID from the perspective of the aux tasks
    - Have to be careful about the relationship between task and dataset (ex: the "0" class in MNIST and Rotation task)




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



## Running Experiments
Different experiment templates can be found in the `experiments` folder.
Note that we use [simple_parsing](https://github.com/lebrice/SimpleParsing) to
make

TODO: This doesn't make much sense (baseline vs baseline_aux, etc etc). Need to figure this out.

- i.i.d setting (baseline)
    ```console
    python main.py baseline
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