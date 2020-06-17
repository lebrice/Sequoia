
from pathlib import Path
import wandb
from simple_parsing import ArgumentParser
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Callable
from utils.utils import add_prefix
import numpy as np

# parser = ArgumentParser(description='Export Wandb results')
# parser.add_arguments(WandbConfig, "config")
# parser.add_arguments(Filter, "filters")
# args = parser.parse_args()
# config: WandbConfig = args.config
# filters: Filter = args.filters
from utils.utils import flatten_dict

# https://docs.wandb.com/library/api/examples#export-metrics-from-all-runs-in-a-project-to-a-csv-file
api = wandb.Api()
import pandas as pd

def get_results(entity: str="lebrice", project: str="SSCL_1", filtering_fn: Callable[[pd.Series], pd.Series]=None, prefix: str="", filters: Dict=None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    filters = filters or {}
    if prefix:
        filters = dict(add_prefix(filters, prefix))
    
    path = f"{entity}/{project}"
    print(path, filters)

    runs = api.runs(path=path, filters=filters)

    print(f"Found {len(runs)} runs for filters {filters}")
    if len(runs) == 0:
        return None, None
    summary_list = [] 
    config_list = [] 
    name_list = [] 
    from tqdm import tqdm

    for run in tqdm(runs): 
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict) 
        # run.config is the input metrics.  We remove special values that start with _.
        config_list.append(add_prefix({k:v for k,v in flatten_dict(run.config, separator=".").items() if not k.startswith('_')}, "config.")) 
        # run.name is the name of the run.
        name_list.append(run.name)

    import pandas as pd 
    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    try:
        if filtering_fn is not None:
            all_df = all_df[filtering_fn(all_df)]
        run_name_key = "run_name"
        if run_name_key not in all_df.columns:
            run_name_key = "name"

        from .make_oml_plot import n_tasks_used
        all_df = all_df.assign(n_aux_tasks=lambda row: list(map(n_tasks_used, row[run_name_key])))
        grouped = all_df.sort_values("n_aux_tasks").groupby([run_name_key], sort=False)
        
        cumul_acc = grouped["Cumul Accuracy [4]"]
        cumul_acc_df = cumul_acc.describe()
        
        knn_acc = grouped["KNN/test/full"]
        knn_acc_df = knn_acc.describe()

    except KeyError as e:
        print(f"MISSING KEY: {e}")
        print(*list(all_df.columns), sep="\n")
        raise e

    return cumul_acc_df, knn_acc_df

def get_mean_std_string(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["count"] > 1][["mean", "std"]]
    return df.assign(s=lambda rows: [f"{r['mean']:.2f} ± {r['std']:.2f}" for name, r in rows.iterrows()])["s"]

table = pd.DataFrame()

project = "SSCL_1"
run_group = "cifar10-sh"
col_name = "Base"
filters = {"config.multihead": False, "config.config.run_group": run_group, "config.config.early_stopping.patience": 0}
cumul_acc, knn_acc = get_results(project=project, filters=filters)
sup_acc = get_mean_std_string(cumul_acc)
knn_acc = get_mean_std_string(knn_acc)
print(sup_acc)
table[col_name] = sup_acc


project = "SSCL_1"
run_group = "cifar10-sh_d"
col_name = "Detached"
filtering_fn = lambda runs: (
    (runs["config.early_stopping.patience"] == 0) &
    (~runs["config.multihead"])
)
filters = {"config.multihead": False, "config.config.run_group": run_group, "config.config.early_stopping.patience": 0}
cumul_acc, knn_acc = get_results(project=project, filtering_fn=filtering_fn, filters=filters)
sup_acc = get_mean_std_string(cumul_acc)
knn_acc = get_mean_std_string(knn_acc)
print(sup_acc)
table[col_name] = sup_acc

project = "SSCL_replay"
run_groups = ["cifar10", "cifar10_d"]
col_names = ["Replay", "Replay + Detached"]

for run_group, col_name in zip(run_groups, col_names):
    filters = {
        "config.multihead": False,
        "config.run_group": run_group,
        "config.early_stopping.patience": 0,
        "config.detach_classifier": "_d" in run_group,
    }
    sup_acc, knn_acc = get_results(project=project, filters=filters)
    sup_acc = get_mean_std_string(cumul_acc)
    knn_acc = get_mean_std_string(knn_acc)
    print(sup_acc)
    table[col_name] = sup_acc

print(table.to_latex().replace("±", "$\\pm$"))
exit()
