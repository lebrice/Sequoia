from simple_parsing import ArgumentParser
from typing import Dict, Type, List
from sequoia.common.config import Config
from sequoia.settings import Setting, Results, Method
from pathlib import Path
from collections import defaultdict
import pandas as pd


from sequoia.settings import PassiveSetting, ActiveSetting


def demo_all_settings(MethodType: Type[Method], datasets: List[str] = ["mnist", "fashionmnist"], **setting_kwargs):
    """ Evaluates the given Method on all its applicable settings.
    
    NOTE: Only evaluates on the mnist/fashion-mnist datasets for this demo.
    """
    # Iterate over all the applicable evaluation settings, using the default
    # options for each setting, and store the results inside this dictionary.
    all_results: Dict[Type[Setting], Dict[str, Results]] = defaultdict(dict)
    
    # Loop over all the types of settings this method is applicable on, i.e.
    # all the nodes in the tree below its target Setting). 
    for setting_type in MethodType.get_applicable_settings():
        # Loop over all the available dataset for each setting:
        for dataset in setting_type.get_available_datasets():
            if datasets and dataset not in datasets:
                print(f"Skipping {setting_type} / {dataset} for now.")
                continue

            if issubclass(setting_type, ActiveSetting):
                print(f"Skipping {setting_type} (not considering RL settings for this demo).")
                continue

            # 1. Create a Method of the provided type, so we start fresh every time.
            method = MethodType()

            # 2. Create the setting
            setting = setting_type(dataset=dataset, **setting_kwargs)
            
            # 3. Apply the method on the setting.
            results: Results = setting.apply(method)
            
            print(f"Results on setting {setting_type}, dataset {dataset}:")
            print(results.summary())

            # Save the results in the dict defined above.
            all_results[setting_type][dataset] = results


    # Create a pandas dataframe with all the results:

    result_df: pd.DataFrame = make_result_dataframe(all_results)

    csv_path = Path(f"examples/results/results_{method.get_name()}.csv")
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(csv_path)
    print(f"Saved dataframe with results to path {csv_path}")

    # BONUS: Display the results in a LaTeX-formatted table!

    latex_table_path = Path(f"examples/results/table_{method.get_name()}.tex")
    caption = f"Results for method {type(method).__name__} settings."
    result_df.to_latex(
        buf=latex_table_path,
        caption=caption,
        na_rep="N/A",
        multicolumn=True,
    )
    print(f"Saved LaTeX table with results to path {latex_table_path}")
    
    return all_results


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


def compare_results(all_results: Dict[Type[Method], Dict[Type[Setting], Dict[str, Results]]]) -> None:
    """Helper function, compares the results of the different methods by
    arranging them in a table (pandas dataframe).
    """
    # Make one huge dictionary that maps from:
    # <method, <setting, <dataset, result>>>
    from .demo_utils import make_comparison_dataframe
    comparison_df = make_comparison_dataframe(all_results)
    
    print("----- All Results -------")
    print(comparison_df)

    csv_path = Path("examples/results/comparison.csv")
    latex_path = Path("examples/results/table_comparison.tex")
    
    comparison_df.to_csv(csv_path)
    print(f"Saved dataframe with results to path {csv_path}")
    
    caption = f"Comparison of different methods on their applicable settings."
    comparison_df.to_latex(
        latex_path,
        caption=caption,
        multicolumn=False,
        multirow=False
    )
    print(f"Saved LaTeX table with results to path {latex_path}")


def make_comparison_dataframe(all_results: Dict[Type[Method], Dict[Type[Setting], Dict[str, Results]]]) -> pd.DataFrame:
    """ Helper function: takes in the dictionary with all the results and
    re-arranges it into a pandas dataframe.
    """ 
    # Get all the method names.
    all_methods: List[Type[Method]] = list(all_results.keys())
    all_method_names: List[str] = [m.get_name() for m in all_methods]

    # Get all the setting names.
    all_settings: List[Type[Setting]] = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        all_settings.extend(setting_to_dataset_to_results.keys())
    all_settings = list(set(all_settings))
    all_setting_names: List[str] = [s.get_name() for s in all_settings]

    # Get all the dataset names.
    all_datasets: List[str] = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        for setting, dataset_to_results in setting_to_dataset_to_results.items():
            all_datasets.extend(dataset_to_results.keys())                
    all_datasets = list(set(all_datasets))
    
    
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

