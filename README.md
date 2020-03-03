# SSCL
Self-Supervised Continual Learning


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