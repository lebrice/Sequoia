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

- Baseline (iid setting)
    ```console
    python main.py baseline
    ```

- Baseline + auxiliary tasks (iid setting)
    ```console
    python main.py baseline_aux
    ```

- TODO: Baseline (class-incremental setting) 
    ```console
    python main.py baseline --class-incremental
    ```

- Baseline + auxiliary tasks (class-incremental setting)
    ```console
    python main.py baseline_aux
    ```