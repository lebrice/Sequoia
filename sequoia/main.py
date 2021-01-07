""" Sequoia - The Research Tree 

Used to run experiments, which consist in applying a Method to a Setting.
"""
import argparse

from simple_parsing import ArgumentParser

import sequoia.methods
from sequoia.experiment import Experiment
from sequoia.methods import all_methods
from sequoia.settings import all_settings
from sequoia.utils import get_logger
from sequoia.experiments import Experiment

logger = get_logger(__file__)

def main():
    logger.debug("Registered Settings: \n" + "\n".join(
        f"- {setting.get_name()}: {setting} ({setting.get_path_to_source_file()})" for setting in all_settings
    ))
    logger.debug("Registered Methods: \n" + "\n".join(
        f"- {method.get_name()}: {method} ({method.get_path_to_source_file()})" for method in all_methods
    ))

    # results = Experiment.main()
    # if results:
    #     print("\n\n EXPERIMENT IS DONE \n\n")
    #     # Experiment didn't crash, show results:
    #     print(f"Results: {results}")
    # return results
    
    parser = ArgumentParser(prog="sequoia", description=__doc__,
                            add_dest_to_option_strings=False)
    
    subparsers = parser.add_subparsers(
        title="command",
        description="Command to execute",
        parser_class=ArgumentParser,
        required=True,
    )

    setting_parser = subparsers.add_parser(
        "setting",
        description="Experimental setting",
        add_dest_to_option_strings=False,
    )

    setting_subparsers = setting_parser.add_subparsers(
        title="setting_choice",
        description="choice of setting?",
        dest="setting",
        required=True,
    )
    for setting in all_settings:
        setting_name = setting.get_name()
        setting_parser: ArgumentParser = setting_subparsers.add_parser(
            setting.get_name(),
            description=setting.__doc__,
            add_dest_to_option_strings=False,
        )
        setting.add_argparse_args(parser=setting_parser, dest="setting")

        method_subparsers = setting_parser.add_subparsers(
            title="method",
            description=f"which method to apply to the {setting_name} Setting."
        )
        for method in setting.get_applicable_methods():
            method_name = method.get_name()
            method_parser: ArgumentParser = method_subparsers.add_parser(
                method_name,
                description=method.__doc__,
                add_dest_to_option_strings=False,
            )
            method.add_argparse_args(parser=method_parser, dest="method")
            # method_parser.add_arguments(method, dest="method")

    method_parser = subparsers.add_parser("method", description="Experimental method.")
    method_subparsers = method_parser.add_subparsers(
        title="method_choice",
        description="choice of method?",
        required=True,
    )
    for method in all_methods:
        method_name = method.get_name()
        method_parser: ArgumentParser = method_subparsers.add_parser(
            method.get_name(),
            description=method.__doc__,
            add_dest_to_option_strings=False,
        )
        method.add_argparse_args(parser=method_parser, dest="method")
        
        setting_subparsers = method_parser.add_subparsers(
            title="setting",
            description=f"which setting to apply to the {method_name} method."
        )
        for setting in method.get_applicable_settings():
            setting_name = setting.get_name()
            setting_parser: ArgumentParser = setting_subparsers.add_parser(
                setting_name,
                description=setting.__doc__,
                add_dest_to_option_strings=False,
            )
            setting.add_argparse_args(parser=setting_parser, dest="setting")
            # setting_parser.add_arguments(setting, dest="setting")
    
    args = parser.parse_args()
        
    
    assert False, args
    
        
    subparsers.add_parser("help", )    
    
    
    results = Experiment.main()
    if results:
        print("\n\n EXPERIMENT IS DONE \n\n")
        # Experiment didn't crash, show results:
        print(f"Results: {results}")


if __name__ == "__main__":
    main()
