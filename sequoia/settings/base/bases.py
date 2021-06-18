""" This module defines the base classes for Settings and Methods.
"""
import json
import traceback
from abc import ABC, abstractmethod
from functools import partial
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import gym
from gym.utils import colorize
from pytorch_lightning import LightningDataModule
from wandb.wandb_run import Run

from sequoia.common import Config, Metrics
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.settings.base.results import Results
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.parseable import Parseable
from sequoia.utils.utils import (
    camel_case,
    compute_identity,
    flatten_dict,
    get_path_to_source_file,
    remove_suffix,
)
import wandb

logger = get_logger(__file__)


class SettingABC:
    """ Abstract base class for a Setting.

    This just shows the minimal API. For more info, see the `Setting` class,
    which is the concrete implementation of this class, and the 'root' of the
    tree.

    Abstract (required) methods:
    - **apply** Applies a given Method on this setting to produce Results.

    "Abstract"-ish (required) class attributes:
    - `Results`: The class of Results that are created when applying a Method on
      this setting.
    - `Observations`: The type of Observations that will be produced  in this
        setting.
    - `Actions`: The type of Actions that are expected from this setting.
    - `Rewards`: The type of Rewards that this setting will (potentially) return
      upon receiving an action from the method.
    """

    Results: ClassVar[Type[Results]] = Results
    Observations: ClassVar[Type[Observations]] = Observations
    Actions: ClassVar[Type[Actions]] = Actions
    Rewards: ClassVar[Type[Rewards]] = Rewards

    @abstractmethod
    def apply(self, method: "Method", config: Config = None) -> "SettingABC.Results":
        """ Applies a Method on this experimental Setting to produce Results.

        Defines the training/evaluation procedure specific to this Setting.

        The training/evaluation loop can be defined however you want, as long as
        it respects the following constraints:

        1.  This method should always return either a float or a Results object
            that indicates the "performance" of this method on this setting.

        2. More importantly: You **have** to make sure that you do not break
            compatibility with more general methods targetting a parent setting!
            It should always be the case that all methods designed for any of
            this Setting's parents should also be applicable via polymorphism,
            i.e., anything that is defined to work on the class `Animal` should
            also work on the class `Cat`!

        3. While not enforced, it is strongly encourged that you define your
            training/evaluation routines at a pretty high level, so that Methods
            that get applied to your Setting can make use of pytorch-lightning's
            `Trainer` & `LightningDataModule` API to be neat and fast.

        Parameters
        ----------
        method : Method
            A Method to apply on this Setting.

        config : Optional[Config]
            Optional configuration object with things like the log dir, the data
            dir, cuda, wandb config, etc. When None, will be parsed from the
            current command-line arguments.

        Returns
        -------
        Results
            An object that is used to measure or quantify the performance of the
            Method on this experimental Setting.
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    @abstractmethod
    def train_dataloader(
        self, *args, **kwargs
    ) -> Environment[Observations, Actions, Rewards]:
        pass

    @abstractmethod
    def val_dataloader(
        self, *args, **kwargs
    ) -> Environment[Observations, Actions, Rewards]:
        pass

    @abstractmethod
    def test_dataloader(
        self, *args, **kwargs
    ) -> Environment[Observations, Actions, Rewards]:
        pass

    @classmethod
    @abstractmethod
    def get_available_datasets(cls) -> Iterable[str]:
        """Returns an iterable of the names of available datasets. """

    # --- Below this are some class attributes and methods related to the Tree. ---

    # These are some "private" class attributes.
    # For any new Setting subclass, it's parent setting.
    _parent: ClassVar[Type["SettingABC"]] = None
    # A list of all the direct children of this setting.
    _children: ClassVar[Set[Type["SettingABC"]]] = set()
    # List of all methods that directly target this Setting.
    _targeted_methods: ClassVar[Set[Type["Method"]]] = set()

    def __init_subclass__(cls, **kwargs):
        """ Called whenever a new subclass of `Setting` is declared. """
        # logger.debug(f"Registering a new setting: {cls.get_name()}")

        # Exceptionally, create this new empty list that will hold all the
        # forthcoming subclasses of this particular new setting.
        cls._children = set()
        cls._targeted_methods = set()
        # Inform the immediate parents in the tree that they have a new child.
        for immediate_parent in cls.get_immediate_parents():
            immediate_parent._children.add(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_applicable_methods(cls) -> List[Type["Method"]]:
        """ Returns all the Methods applicable on this Setting. """
        applicable_methods: List[Method] = []
        from sequoia.methods import all_methods

        for method_type in all_methods:
            if method_type.is_applicable(cls):
                applicable_methods.append(method_type)
        return applicable_methods

    @classmethod
    def register_method(cls, method: Type["Method"]):
        """ Register a method as being Applicable on this type of Setting. """
        cls._targeted_methods.add(method)

    @classmethod
    def get_name(cls) -> str:
        """ Gets the name of this Setting. """
        # LightningDataModule has a `name` class attribute of `...`!
        if getattr(cls, "name", None) != Ellipsis:
            return cls.name
        name = camel_case(cls.__qualname__)
        return remove_suffix(name, "_setting")

    @classmethod
    def immediate_children(cls) -> Iterable[Type["SettingABC"]]:
        """ Returns the immediate children of this Setting in the hierarchy.
        In most cases, this will be a list with only one value.
        """
        yield from cls._children

    @classmethod
    def get_immediate_children(cls) -> List[Type["SettingABC"]]:
        """ Returns a list of the immediate children of this Setting. """
        return list(cls.immediate_children())

    @classmethod
    def children(cls) -> Iterable[Type["SettingABC"]]:
        """Returns an Iterator over all the children of this Setting, in-order.
        """
        # Yield the immediate children.
        for child in cls._children:
            yield child
            # Yield from the children themselves.
            yield from child.children()

    @classmethod
    def get_children(cls) -> List[Type["SettingABC"]]:
        return list(cls.children())

    @classmethod
    def immediate_parents(cls) -> List[Type["SettingABC"]]:
        """ Returns the immediate parent(s) Setting(s).
        In most cases, this will be a list with only one value.
        """
        return [parent for parent in cls.__bases__ if issubclass(parent, SettingABC)]

    @classmethod
    def get_immediate_parents(cls) -> List[Type["SettingABC"]]:
        """ Returns the immediate parent(s) Setting(s).
        In most cases, this will be a list with only one value.
        """
        return cls.immediate_parents()

    @classmethod
    def parents(cls) -> Iterable[Type["SettingABC"]]:
        """yields the lineage, from bottom to top.

        NOTE: In the case of Settings having multiple parents (such as TraditionalSLSetting),
        this is still just a list that reflects the method resolution order for that
        setting.
        """
        return [
            parent_class
            for parent_class in cls.mro()[1:]
            if issubclass(parent_class, SettingABC)
        ]

    @classmethod
    def get_parents(cls) -> List[Type["SettingABC"]]:
        return list(cls.parents())

    @classmethod
    def get_path_to_source_file(cls: Type) -> Path:
        from sequoia.utils.utils import get_path_to_source_file

        return get_path_to_source_file(cls)

    @classmethod
    def get_tree_string(
        cls,
        formatting: str = "command_line",
        with_methods: bool = False,
        with_assumptions: bool = False,
        with_docstrings: bool = False,
    ) -> str:
        """ Returns a string representation of the tree starting at this node downwards.
        """
        from sequoia.utils.readme import get_tree_string, get_tree_string_markdown

        formatting_functions = {
            "command_line": get_tree_string,
            "markdown": get_tree_string_markdown,
        }
        if formatting not in formatting_functions.keys():
            raise RuntimeError(
                f"formatting must be one of {','.join(formatting_functions)}, "
                f"got {formatting}"
            )
        return formatting_functions[formatting](
            cls,
            with_methods=with_methods,
            with_assumptions=with_assumptions,
            with_docstrings=with_docstrings,
        )


SettingType = TypeVar("SettingType", bound=SettingABC)


class Method(Generic[SettingType], Parseable, ABC):
    """ ABC for a Method, which is a solution to a research problem (a Setting).
    """

    # Class attribute that holds the setting this method was designed to target.
    # Needs to either be passed to the class statement or set as a class
    # attribute.
    target_setting: ClassVar[Type[SettingType]] = None

    _training: bool

    def configure(self, setting: SettingType) -> None:
        """Configures this method before it gets applied on the given Setting.

        Args:
            setting (SettingType): The setting the method will be evaluated on.
        """

    @abstractmethod
    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Union[Actions, Any]:
        """ Get a batch of predictions (actions) for the given observations.
        returned actions must fit the action space.
        """

    @abstractmethod
    def fit(
        self,
        train_env: Environment[Observations, Actions, Rewards],
        valid_env: Environment[Observations, Actions, Rewards],
    ):
        """Called by the Setting to give the method data to train with.

        Might be called more than once before training is 'complete'.
        """

    def test(self, test_env: Environment[Observations, Actions, Optional[Rewards]]):
        """ (WIP) Optional method which could be called by the setting to give
        your Method more flexibility about how it wants to arrange the test env.

        Parameters
        ----------
        test_env : Environment[Observations, Actions, Optional[Rewards]]
            Test environment which monitors your actions, and in which you are
            only allowed a limited number of steps.
        """
        raise NotImplementedError

    def receive_results(self, setting: SettingType, results: Results) -> None:
        """ Receive the Results of applying this method on the given Setting.

        This method is optional.

        This will be called after the method has been successfully applied to
        a Setting, and could be used to log or persist the results somehow.

        Parameters
        ----------
        results : Results
            The `Results` object constructed by `setting`, as a result of applying
            this Method to it.
        """
        
        run_name = ""
        # Set the default name for this run.
        # run_name = f"{method_name}-{setting_name}"
        # dataset = getattr(self, "dataset", None)
        # if isinstance(dataset, str):
        #     run_name += f"-{dataset}"
        # if getattr(self, "nb_tasks", 0) > 1:
        #     run_name += f"_{self.nb_tasks}t"

        setting_name = setting.get_name()
        method_name = self.get_name()
        base_results_dir: Path = setting.config.log_dir / setting_name / method_name
        
        dataset_name = getattr(setting, "dataset", None)
        if isinstance(dataset_name, str):
            base_results_dir /= dataset_name
        
        if wandb.run and wandb.run.id:
            # if setting.wandb and setting.wandb.project:
            run_id = wandb.run.id
            assert isinstance(run_id, str)
            # results_dir = base_results_dir / run_id
            # TODO: Fix this:
            results_dir = wandb.run.dir
        else:
            for suffix in [f"run_{i}"  for i in range(100)]:
                results_dir = base_results_dir / suffix
                try:
                    results_dir.mkdir(exist_ok=False, parents=True)
                except FileExistsError:
                    pass
                else:
                    break
            else:
                raise RuntimeError(
                    f"Unable to create a unique results dir under {base_results_dir} "
                )
        results_dir = Path(results_dir)
        logger.info(f"Saving results in directory {results_dir}")
        results_json_path = results_dir / "results.json"
        try:
            with open(results_json_path, "w") as f:
                json.dump(results.to_log_dict(), f)
        except Exception as e:
            print(f"Unable to save the results: {e}")

        setting_path = results_dir / "setting.yaml"
        try:
            setting.save(setting_path)
        except Exception as e:
            print(f"Unable to save the Setting: {e}")

        method_path = results_dir / "method.yaml"
        try:
            self.save(method_path)
        except Exception as e:
            print(f"Unable to save the Method: {e}")

        if wandb.run:
            wandb.save(str(results_json_path))
            if setting_path.exists():
                wandb.save(str(setting_path))
            if method_path.exists():
                wandb.save(str(method_path))

    def setup_wandb(self, run: Run) -> None:
        """ Called by the Setting when using Weights & Biases, after `wandb.init`.

        This method is here to provide Methods with the opportunity to log some of their
        configuration options or hyper-parameters to wandb.

        NOTE: The Setting has already set the `"setting"` entry in the `wandb.config` by
        this point.

        Parameters
        ----------
        run : wandb.Run
            Current wandb Run.
        """

    def set_training(self) -> None:
        """Called by the Setting to let the Method know it is in the "training" phase.

        By default, this will try to to look for any nn.Module attributes on `self`, and
        call their `train()` method.
        """
        self._training = True
        try:
            from torch import nn

            for attribute, value in vars(self).items():
                if isinstance(value, nn.Module):
                    logger.debug(
                        f"Calling 'train()' on the Method's {attribute} attribute."
                    )
                    value.train()
        except Exception as exc:
            logger.warning(
                f"Unable to call `train()` on nn.Modules of the Method: {exc}"
            )

    def set_testing(self) -> None:
        """Called by the Setting to let the Method know when it is in "testing" phase.

        By default, this will try to to look for any nn.Module attributes on `self`, and
        call their `eval()` method.
        """
        self._training = False
        try:
            from torch import nn

            for attribute, value in vars(self).items():
                if isinstance(value, nn.Module):
                    logger.debug(
                        f"Calling 'eval()' on the Method's {attribute} attribute."
                    )
                    value.eval()
        except Exception as exc:
            logger.warning(
                f"Unable to call `eval()` on nn.Modules of the Method: {exc}"
            )

    @property
    def training(self) -> bool:
        """Wether we're currently in the 'training' phase.

        Returns
        -------
        bool
            Wether we're in the 'training' phase or not.
        """
        return getattr(self, "_training", True)

    @property
    def testing(self) -> bool:
        """Wether we're currently in the 'testing' phase.

        Returns
        -------
        bool
            Wether we're in the 'testing' phase or not.
        """
        return not self.training

    # --------
    # Below this are some class attributes and methods related to the Tree
    # structure and for launching Experiments using this method.
    # --------

    @classmethod
    def main(cls, argv: Optional[Union[str, List[str]]] = None) -> Results:
        """ Run an Experiment from the command-line using this method.

        (TODO: @lebrice Finish writing a good docstring here that explains how this works
        and how to use it.)
        You can then select which setting, dataset, etc. this method will be
        applied to using the --setting <setting_name>, and the rest of the
        arguments will be passed to the Setting's from_args method.
        """

        from sequoia.main import Experiment

        experiment: Experiment
        # Create the Method object from the command-line:
        method = cls.from_args(argv, strict=False)
        # Then create the 'Experiment' from the command-line, which makes it
        # possible to choose between all the settings.
        experiment = Experiment.from_args(argv, strict=False)
        # Set the method attribute to be the one parsed above.
        experiment.method = method
        results: Results = experiment.launch(argv)
        return results

    @classmethod
    def is_applicable(cls, setting: Union[SettingType, Type[SettingType]]) -> bool:
        """Returns wether this Method is applicable to the given setting.

        A method is applicable on a given setting if and only if the setting is
        the method's target setting, or if it is a descendant of the method's
        target setting (below the target setting in the tree).

        Concretely, since the tree is implemented as an inheritance hierarchy,
        a method is applicable to any setting which is an instance (or subclass)
        of its target setting.

        Args:
            setting (SettingABC): a Setting.

        Returns:
            bool: Wether or not this method is applicable on the given setting.
        """

        # if given an object, get it's type.
        if isinstance(setting, LightningDataModule):
            setting = type(setting)

        if not issubclass(setting, SettingABC) and issubclass(
            setting, LightningDataModule
        ):
            # TODO: If we're trying to check if this method would be compatible
            # with a LightningDataModule, rather than a Setting, then we treat
            # that LightningModule the same way we would an TraditionalSLSetting.
            # i.e., if we're trying to apply a Method on something that isn't in
            # the tree, then we consider that datamodule as the TraditionalSLSetting node.
            from sequoia.settings import TraditionalSLSetting

            setting = TraditionalSLSetting

        return issubclass(setting, cls.target_setting)

    @classmethod
    def get_applicable_settings(cls) -> List[Type[SettingType]]:
        """ Returns all settings on which this method is applicable.
        NOTE: This only returns 'concrete' Settings.
        """
        from sequoia.settings import all_settings

        return list(filter(cls.is_applicable, all_settings))
        # This would return ALL the setting:
        # return list([cls.target_setting, *cls.target_setting.children()])

    @classmethod
    def all_evaluation_settings(cls, **kwargs) -> Iterable[SettingType]:
        """ Generator over all the combinations of Settings/datasets on which
        this method is applicable.

        If keyword arguments are passed, they will be passed to the constructor
        of each setting.
        """
        for setting_type in cls.get_applicable_settings():
            for dataset in setting_type.get_available_datasets():
                setting = setting_type(dataset=dataset, **kwargs)
                yield setting

    @classmethod
    def get_name(cls) -> str:
        """ Gets the name of this method class. """
        name = getattr(cls, "name", None)
        if name is None:
            name = camel_case(cls.__qualname__)
            name = remove_suffix(name, "_method")
        return name

    @classmethod
    def get_family(cls) -> str:
        """ Gets the name of the 'family' of Methods which contains this method class.

        This is used to differentiate methods with the same name, for instance
        sb3/DQN versus pl_bolts/DQN, sequoia/EWC vs avalanche/EWC, etc.
        """
        family = getattr(cls, "family", None)
        if family is None:
            # Use the name of the parent folder as the 'family'.
            method_source_path = cls.get_path_to_source_file()
            family = method_source_path.parent.name
        return family

    def __init_subclass__(
        cls, target_setting: Type[SettingType] = None, **kwargs
    ) -> None:
        """Called when creating a new subclass of Method.

        Args:
            target_setting (Type[Setting], optional): The target setting.
                Defaults to None, in which case the method will inherit the
                target setting of it's parent class.
        """
        if target_setting:
            cls.target_setting = target_setting
        elif getattr(cls, "target_setting", None):
            target_setting = cls.target_setting
        else:
            raise RuntimeError(
                f"You must either pass a `target_setting` argument to the "
                f"class statement or have a `target_setting` class variable "
                f"when creating a new subclass of {__class__}."
            )
        # Register this new method on the Setting.
        target_setting.register_method(cls)
        return super().__init_subclass__(**kwargs)

    @classmethod
    def get_path_to_source_file(cls) -> Path:
        return get_path_to_source_file(cls)

    def get_experiment_name(
        self, setting: SettingABC, experiment_id: str = None
    ) -> str:
        """Gets a unique name for the experiment where `self` is applied to `setting`.

        This experiment name will be passed to `orion` when performing a run of
        Hyper-Parameter Optimization.

        Parameters
        ----------
        - setting : Setting

            The `Setting` onto which this method will be applied. This method will be used when

        - experiment_id: str, optional

            A custom hash to append to the experiment name. When `None` (default), a
            unique hash will be created based on the values of the Setting's fields.

        Returns
        -------
        str
            The name for the experiment.
        """
        if not experiment_id:
            setting_dict = setting.to_dict()
            # BUG: Some settings have non-string keys/value or something?
            d = flatten_dict(setting_dict)
            experiment_id = compute_identity(size=5, **d)
        assert isinstance(
            setting.dataset, str
        ), "assuming that dataset is a str for now."
        return (
            f"{self.get_name()}-{setting.get_name()}_{setting.dataset}_{experiment_id}"
        )

    def get_search_space(self, setting: SettingABC) -> Mapping[str, Union[str, Dict]]:
        """Returns the search space to use for HPO in the given Setting.

        Parameters
        ----------
        setting : Setting
            The Setting on which the run of HPO will take place.

        Returns
        -------
        Mapping[str, Union[str, Dict]]
            An orion-formatted search space dictionary, mapping from hyper-parameter
            names (str) to their priors (str), or to nested dicts of the same form.
        """
        raise NotImplementedError(
            "You need to provide an implementation for the `get_search_space` method "
            "in order to enable HPO sweeps."
        )

    def adapt_to_new_hparams(self, new_hparams: Dict[str, Any]) -> None:
        """Adapts the Method when it receives new Hyper-Parameters to try for a new run.

        It is required that this method be implemented if you want to perform HPO sweeps
        with Orion.

        NOTE: It is very strongly recommended that you always re-create your model and
        any modules / components that depend on these hyper-parameters inside the
        `configure` method! (Otherwise these new hyper-parameters will not be used in
        the next run)

        Parameters
        ----------
        new_hparams : Dict[str, Any]
            The new hyper-parameters being recommended by the HPO algorithm. These will
            have the same structure as the search space.
        """
        raise NotImplementedError(
            "You need to provide an implementation for the `adapt_to_new_hparams` "
            "method in order to enable HPO sweeps."
        )

    def hparam_sweep(
        self,
        setting: SettingABC,
        search_space: Dict[str, Union[str, Dict]] = None,
        experiment_id: str = None,
        database_path: Union[str, Path] = None,
        max_runs: int = None,
        debug: bool = False,
    ) -> Tuple[Dict, float]:
        """ Performs a Hyper-Parameter Optimization sweep using orion.

        Changes the values in `self.hparams` iteratively, returning the best hparams
        found so far.

        Parameters
        ----------
        setting : Setting
            Setting to run the sweep on.

        search_space : Dict[str, Union[str, Dict]], optional
            Search space of the hyper-parameter optimization algorithm. Defaults to
            `None`, in which case the result of the `get_search_space` method is used.

        experiment_id : str, optional
            Unique Id to use when creating the experiment in Orion. Defaults to `None`,
            in which case a hash of the `setting`'s fields is used.

        database_path : Union[str, Path], optional
            Path to a pickle file to be used by Orion to store the hyper-parameters and
            their corresponding values. Default to `None`, in which case the database is
            created at path `./orion_db.pkl`.

        max_runs : int, optional
            Maximum number of runs to perform. Defaults to `None`, in which case the run
            lasts until the search space is exhausted.

        debug : bool, optional
            Wether to run Orion in debug-mode, where the database is an EphemeralDb,
            meaning it gets created for the sweep and destroyed at the end of the sweep.

        Returns
        -------
        Tuple[BaselineModel.HParams, float]
            Best HParams, and the corresponding performance.
        """
        try:
            from orion.client import build_experiment
            from orion.core.worker.trial import Trial
        except ImportError as e:
            raise RuntimeError(
                f"Need to install the optional dependencies for HPO, using "
                f"`pip install -e .[hpo]` (error: {e})"
            ) from e

        search_space = search_space or self.get_search_space(setting)
        logger.info("HPO Search space:\n" + json.dumps(search_space, indent="\t"))

        database_path: Path = Path(database_path or "./orion_db.pkl")
        logger.info(f"Will use database at path '{database_path}'.")
        experiment_name = self.get_experiment_name(setting, experiment_id=experiment_id)

        experiment = build_experiment(
            name=experiment_name,
            space=search_space,
            debug=debug,
            algorithms="BayesianOptimizer",
            max_trials=max_runs,
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": str(database_path)},
            },
        )

        previous_trials: List[Trial] = experiment.fetch_trials_by_status("completed")
        # Since Orion works in a 'lower is better' fashion, so if the `objective` of the
        # Results class for the given Setting have "higher is better", we negate the
        # objectives when extracting them and again before submitting them to Orion.
        lower_is_better = setting.Results.lower_is_better
        sign = 1 if lower_is_better else -1
        if previous_trials:
            logger.info(
                f"Using existing Experiment {experiment} which has "
                f"{len(previous_trials)} existing trials."
            )
        else:
            logger.info(f"Created new experiment with name {experiment_name}")

        trials_performed = 0
        failed_trials = 0

        red = partial(colorize, color="red")
        green = partial(colorize, color="green")

        while not (experiment.is_done or failed_trials == 3):
            # Get a new suggestion of hparams to try:
            trial: Trial = experiment.suggest()

            # ---------
            # (Re)create the Model with the suggested Hparams values.
            # ---------

            new_hparams: Dict = trial.params
            # Inner function, just used to make the code below a bit simpler.
            # TODO: We should probably also change some values in the Config (e.g.
            # log_dir, checkpoint_dir, etc) between runs.
            logger.info(
                "Suggested values for this run:\n"
                + json.dumps(new_hparams, indent="\t")
            )
            self.adapt_to_new_hparams(new_hparams)

            # ---------
            # Evaluate the (adapted) method on the setting:
            # ---------
            try:
                result: Results = setting.apply(self)
            except Exception:

                logger.error(red("Encountered an error, this trial will be dropped:"))
                logger.error(red("-" * 60))
                with StringIO() as s:
                    traceback.print_exc(file=s)
                    s.seek(0)
                    logger.error(red(s.read()))
                logger.error(red("-" * 60))
                failed_trials += 1
                logger.error(red(f"({failed_trials} failed trials so far). "))

                experiment.release(trial)
            else:
                # Report the results to Orion:
                orion_result = dict(
                    name=result.objective_name,
                    type="objective",
                    value=sign * result.objective,
                )
                experiment.observe(trial, [orion_result])
                trials_performed += 1
                logger.info(
                    green(
                        f"Trial #{trials_performed}: {result.objective_name} = {result.objective}"
                    )
                )
                # Receive the results, maybe log to wandb, whatever you wanna do.
                self.receive_results(setting, result)

        logger.info(
            "Experiment statistics: \n"
            + "\n".join(f"\t{key}: {value}" for key, value in experiment.stats.items())
        )
        logger.info(f"Number of previous trials: {len(previous_trials)}")
        logger.info(f"Trials successfully completed by this worker: {trials_performed}")
        logger.info(f"Failed Trials attempted by this worker: {failed_trials}")

        if "best_trials_id" not in experiment.stats:
            raise RuntimeError("Can't find the best trial, experiment might be broken!")

        best_trial: Trial = experiment.get_trial(uid=experiment.stats["best_trials_id"])
        best_hparams = best_trial.params
        best_objective = best_trial.objective
        return best_hparams, best_objective
