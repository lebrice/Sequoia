"""Runs an experiment, which consist in applying a Method to a Setting.


"""
import methods
from sequoia.methods import all_methods
from sequoia.settings import all_settings
from sequoia.utils import get_logger

from experiment import Experiment

logger = get_logger(__file__)

if __name__ == "__main__":        
    logger.debug("Registered Settings: \n" + "\n".join(
        f"- {setting.get_name()}: {setting} ({setting.get_path_to_source_file()})" for setting in all_settings
    ))
    logger.debug("Registered Methods: \n" + "\n".join(
        f"- {method.get_name()}: {method} ({method.get_path_to_source_file()})" for method in all_methods
    ))

    results = Experiment.main()
    if results:
        print("\n\n EXPERIMENT IS DONE \n\n")
        # Experiment didn't crash, show results:
        print(f"Results: {results}")
