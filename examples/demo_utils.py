from simple_parsing import ArgumentParser
from typing import Dict, Type, List
from common.config import Config
from settings import Setting, Results, MethodABC
from pathlib import Path

import pandas as pd

def demo(method_type: Type[MethodABC]) -> Dict[Type[Setting], Dict[str, Results]]:
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Config, dest="config", default=Config())
    parser.add_arguments(method_type.HParams, dest="hparams")

    args = parser.parse_args()

    config: Config = args.config
    hparams: method_type.HParams = args.hparams
    method = method_type(hparams=hparams, config=config)

    all_results: Dict[Type[Setting], Dict[str, Results]] = {}
    
    for SettingClass in method.get_applicable_settings():
        all_results[SettingClass] = {}
        
        # for dataset in SettingClass.available_datasets:
        for dataset in ["mnist", "fashion_mnist"]:
            # Instantiate the Setting, using the default options for each
            # setting, for instance the number of tasks, etc.
            setting = SettingClass(dataset=dataset)
            # Apply the method on the setting.
            results: Results = setting.apply(method, config=config)
            all_results[SettingClass][dataset] = results
            
            ## Use this (and comment out below) to debug just the tables below:
            # class FakeResult:
            #     objective: float = 1.23
            # all_results[SettingClass][dataset] = FakeResult()
             
            print(f"Results for Method {method.get_name()} on setting {SettingClass}, dataset {dataset}:")
            print(results.summary())
    return all_results
    print(f"----- All Results for method {method_type} -------")


def make_result_dataframe(all_results):
    # Create a LaTeX table with all the results for all the settings.
    import pandas as pd
    
    all_settings: List[Type[Setting]] = list(all_results.keys())
    all_setting_names: List[str] = [s.get_name() for s in all_settings]

    all_datasets: List[str] = []
    for setting, dataset_to_results in all_results.items():
        all_datasets.extend(dataset_to_results.keys())                
    all_datasets = list(set(all_datasets))
    
    ## Create a multi-index for the dataframe.
    # tuples = []
    # for setting, dataset_to_results in all_results.items():
    #     setting_name = setting.get_name()
    #     tuples.extend((setting_name, dataset) for dataset in dataset_to_results.keys())
    # tuples = sorted(list(set(tuples)))
    # multi_index = pd.MultiIndex.from_tuples(tuples, names=["setting", "dataset"])
    # single_index = pd.Index(["Objective"])
    # df = pd.DataFrame(index=multi_index, columns=single_index)

    df = pd.DataFrame(index=all_setting_names, columns=all_datasets)

    for setting_type, dataset_to_results in all_results.items():
        setting_name = setting_type.get_name()
        for dataset, result in dataset_to_results.items():
            # df["Objective"][setting_name, dataset] = result.objective
            df[dataset][setting_name] = result.objective
    return df


def save_results_table(result_df: pd.DataFrame, csv_path: Path, latex_path: Path, caption: str = None): 
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(csv_path, "w") as f:
        result_df.to_csv(f)
    print(f"Saved dataframe with results to path {csv_path}")
    result_df.to_latex(
        buf=latex_path,
        caption=caption,
        na_rep="N/A",
        multicolumn=True,
    )
    print(f"Saved LaTeX table with results to path {latex_path}")

