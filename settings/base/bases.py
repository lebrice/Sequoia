import dataclasses
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (Any, ClassVar, Generic, Iterable, List, Optional, Set,
                    Type, TypeVar, Union)

import gym
import numpy as np
from pytorch_lightning import LightningDataModule
from utils.logging_utils import get_logger
from utils.utils import get_path_to_source_file

from common import Config, Metrics
from settings.base.environment import (Actions, Environment, Observations,
                                       Rewards)
from settings.base.objects import Actions, Observations, Rewards
from settings.base.results import Results
from utils.utils import camel_case, remove_suffix

logger = get_logger(__file__)


class SettingABC(LightningDataModule):
    """ Abstract base class for a Setting.

    This just shows the minimal API. For more info, see the `Setting` class,
    which is the concrete implementation of this class, and the 'root' of the
    tree.

    Abstract (required) methods:
    - (new) **apply** Applies a given Method on this setting to produce Results.
    - **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    - **setup**  (things to do on every accelerator in distributed mode).
    - **train_dataloader** the training environment/dataloader.
    - **val_dataloader** the val environments/dataloader(s).
    - **test_dataloader** the test environments/dataloader(s).
    
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
    def apply(self, method: "MethodABC", config: Config = None) -> "SettingABC.Results":
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
        # For illustration purposes only:
        self.config = config or Config.from_args()
        method.configure(self)
        # IDEA: Maybe instead of passing the train_dataloader or test_dataloader,
        # objects, we could instead pass the methods of the LightningDataModule
        # themselves, so we wouldn't have to configure the 'batch_size' etc
        # arguments, and this way we could also directly control how many times
        # the dataloader method can be called, for instance to limit the number
        # of samples that a user can have access to (the number of epochs, etc).
        # Or the dataloader would only allow a given number of iterations!
        method.fit(
            train_env=self.train_dataloader(),
            valid_env=self.val_dataloader(),
        )
        
        test_metrics = []
        test_environment = self.test_dataloader()
        for observations in test_environment:
            # Get the predictions/actions:
            actions = method.get_actions(observations, test_environment.action_space)
            # Get the rewards for the given predictions.
            rewards = test_environment.send(actions)
            # Calculate the 'metrics' (TODO: This should be done be in the env!)
            metrics = self.get_metrics(actions=actions, rewards=rewards)
            test_metrics.append(metrics)

        results = self.Results(test_metrics)
        # TODO: allow the method to observe a 'copy' of the results, maybe?
        method.receive_results(self, results)
        return results

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    @abstractmethod
    def train_dataloader(self, *args, **kwargs) -> Environment[Observations, Actions, Rewards]:
        pass

    @abstractmethod
    def val_dataloader(self, *args, **kwargs) -> Environment[Observations, Actions, Rewards]:
        pass

    @abstractmethod
    def test_dataloader(self, *args, **kwargs) -> Environment[Observations, Actions, Rewards]:
        pass

    @abstractmethod
    def get_metrics(self,
                    actions: Actions,
                    rewards: Rewards) -> Union[float, Metrics]:
        """ Calculate the "metric" from the model predictions (actions) and the
        true labels (rewards).
        """

    @classmethod
    @abstractmethod
    def get_available_datasets(cls) -> Iterable[str]:
        """Returns an iterable of the names of available datasets. """

    ## Below this are some class attributes and methods related to the Tree.

    # These are some "private" class attributes.
    # For any new Setting subclass, it's parent setting.
    _parent: ClassVar[Type["Setting"]] = None
    # A list of all the direct children of this setting.
    _children: ClassVar[Set[Type["SettingABC"]]] = set()
    # List of all methods that directly target this Setting.
    _targeted_methods: ClassVar[Set[Type["MethodABC"]]] = set()
    
    def __init_subclass__(cls, **kwargs):
        """ Called whenever a new subclass of `Setting` is declared. """
        logger.debug(f"Registering a new setting: {cls.get_name()}")

        # Exceptionally, create this new empty list that will hold all the
        # forthcoming subclasses of this particular new setting.
        cls._children = set()
        cls._targeted_methods = set()
        # Inform the immediate parent in the tree, telling it that it has a new
        # child.
        parent = cls.get_parent()
        parent._children.add(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_applicable_methods(cls) -> List[Type["MethodABC"]]:
        """ Returns all the Methods applicable on this Setting. """
        applicable_methods: List[MethodABC] = []
        applicable_methods.extend(cls._targeted_methods)
        if cls._parent:
            applicable_methods.extend(cls._parent.get_applicable_methods())
        return applicable_methods

    @classmethod
    def register_method(cls, method: Type["MethodABC"]):
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
    def get_children(cls) -> List[Type["SettingABC"]]:
        return cls._children

    @classmethod
    def all_children(cls) -> Iterable[Type["SettingABC"]]:
        """Iterates over all the children of this Setting, in-order.
        """
        # Yield the immediate children.
        for child in cls._children:
            yield child
            # Yield from the children themselves.
            yield from child.all_children()

    @classmethod
    def get_parent(cls) -> Optional[Type["SettingABC"]]:
        """Returns the first base class that is an instance of SettingABC, else
        None
        """
        base_nodes = [
            base for base in cls.__bases__
            if inspect.isclass(base) and issubclass(base, SettingABC)
        ]
        return base_nodes[0] if base_nodes else None

    @classmethod
    def get_parents(cls) -> Iterable[Type["SettingABC"]]:
        """TODO: yields the lineage, from bottom to top. """
        parent = cls.get_parent()
        if parent:
            yield parent
            yield from parent.get_parents()


SettingType = TypeVar("SettingType", bound=SettingABC)


class MethodABC(Generic[SettingType], ABC):
    """ ABC for a Method, which is a solution to a research problem (a Setting).
    """
    # Class attribute that holds the setting this method was designed to target.
    # Needs to either be passed to the class statement or set as a class
    # attribute.
    target_setting: ClassVar[Type[SettingType]] = None

    def configure(self, setting: SettingType) -> None:
        """Configures the method before it gets applied on the given Setting.

        Args:
            setting (SettingType): The setting the method will be evaluated on.
        
        TODO: This might be a problem if we're gonna avoid 'cheating'.. we're
        essentially giving the 'Setting' object
        directly to the method.. so I guess the object could maybe 
        """
    
    @abstractmethod
    def get_actions(self, observations: Observations, action_space: gym.Space) -> Union[Actions, Any]:
        """ Get a batch of predictions (actions) for the given observations.
        returned actions must fit the action space.
        """

    @abstractmethod
    def fit(self,
            train_env: Environment[Observations, Actions, Rewards] = None,
            valid_env: Environment[Observations, Actions, Rewards] = None,
            datamodule: LightningDataModule = None):
        """Called by the Setting to give the method data to train with.

        Might be called more than once before training is 'complete'.
        """

    def receive_results(self, setting: SettingType, results: Results) -> None:
        """ Receive the Results of applying this method on the given Setting.
        
        This will be called after the method has been successfully applied to
        a Setting, and could be used to log or persist the results somehow.

        Parameters
        ----------
        results : Results
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """

    ## Below this are some class attributes and methods related to the Tree.

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
        
        if (not issubclass(setting, SettingABC)
            and issubclass(setting, LightningDataModule)):
            # TODO: If we're trying to check if this method would be compatible
            # with a LightningDataModule, rather than a Setting, then we treat
            # that LightningModule the same way we would an IIDSetting.
            # i.e., if we're trying to apply a Method on something that isn't in
            # the tree, then we consider that datamodule as the IIDSetting node.
            from settings import IIDSetting
            setting = IIDSetting

        return issubclass(setting, cls.target_setting)

    @classmethod
    def get_applicable_settings(cls) -> List[Type[SettingType]]:
        """ Returns all settings on which this method is applicable.
        NOTE: This only returns 'concrete' Settings.
        """
        from settings import all_settings
        return list(filter(cls.is_applicable, all_settings))
        # This would return ALL the setting:
        # return list([cls.target_setting, *cls.target_setting.all_children()])

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

    def __init_subclass__(cls, target_setting: Type[SettingType] = None, **kwargs) -> None:
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
