from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Type

from simple_parsing import ArgumentParser

from methods import BaselineMethod
from settings import ClassIncrementalSetting, MethodABC, Results, Setting

from .demo_utils import make_result_dataframe


@dataclass
class MinimalDemoMethod(BaselineMethod, target_setting=ClassIncrementalSetting):
    pass



def demo(method_type=MinimalDemoMethod):
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(method_type, dest="method")
    args = parser.parse_args()
    method = args.method
    
    all_results: Dict[Type[Setting], Dict[str, Results]] = {}
    for SettingClass in method.get_applicable_settings():
        all_results[SettingClass] = {}
        
        for dataset in ["mnist", "fashion_mnist"]:
            setting = SettingClass(dataset=dataset)
            results: Results = setting.apply(method, config=method.config)
            all_results[SettingClass][dataset] = results
            
            print(f"Results for Method {method.get_name()} on setting {SettingClass}, dataset {dataset}:")
            print(results.summary())

    result_df = make_result_dataframe(all_results)
    
    csv_path = Path(f"examples/results/results_{method_type.get_name()}.csv")
    latex_table_path = Path(f"examples/results/table_{method_type.get_name()}.tex")

    caption = f"Results for method {method_type.__name__} on all its applicable settings."
    
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(csv_path)
    print(f"Saved dataframe with results to path {csv_path}")
    result_df.to_latex(
        buf=latex_table_path,
        caption=caption,
        na_rep="N/A",
        multicolumn=True,
    )
    print(f"Saved LaTeX table with results to path {latex_table_path}")
    
if __name__ == "__main__":
    demo()