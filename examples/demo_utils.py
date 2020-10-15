from simple_parsing import ArgumentParser
from typing import Dict, Type, List
from common.config import Config
from settings import Setting, Results, Method
from pathlib import Path

import pandas as pd



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


def make_comparison_dataframe(all_results: Dict[Type[Method], Dict[Type[Setting], Dict[str, Results]]]) -> pd.DataFrame:
    all_methods: List[Type[Method]] = list(all_results.keys())
    all_method_names: List[str] = [m.get_name() for m in all_methods]

    all_settings: List[Type[Setting]] = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        all_settings.extend(setting_to_dataset_to_results.keys())
    all_settings = list(set(all_settings))
    all_setting_names: List[str] = [s.get_name() for s in all_settings]

    all_datasets: List[str] = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        for setting, dataset_to_results in setting_to_dataset_to_results.items():
            all_datasets.extend(dataset_to_results.keys())                
    all_datasets = list(set(all_datasets))
    
    import pandas as pd
    
    # Create the a multi-index, so we can later index df[setting, datset][method]
    # Option 1: All [settings x all datasets]
    # iterables = [all_setting_names, all_datasets]
    # columns = pd.MultiIndex.from_product(iterables, names=["setting", "dataset"])

    # Option 2: Index will be [Setting, <datasets in that setting>]
    # Create the column index using the tuples that apply.
    tuples = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        for setting, dataset_to_results in setting_to_dataset_to_results.items():
            setting_name = setting.get_name()
            tuples.extend((setting_name, dataset) for dataset in dataset_to_results.keys())
    tuples = sorted(list(set(tuples)))          
    multi_index = pd.MultiIndex.from_tuples(tuples, names=["setting", "dataset"])
    single_index = pd.Index(all_method_names, name="Method")
    
    df = pd.DataFrame(index=multi_index, columns=single_index)

    for method_class, setting_to_dataset_to_results in all_results.items():
        method_name = method_class.get_name()
        for setting, dataset_to_results in setting_to_dataset_to_results.items():
            setting_name = setting.get_name()
            for dataset, result in dataset_to_results.items():
                df[method_name][setting_name, dataset] = result.objective
    return df

