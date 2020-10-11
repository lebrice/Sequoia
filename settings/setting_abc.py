import dataclasses
import inspect
from abc import ABC, abstractmethod
from typing import (ClassVar, Iterable, List, Optional, Set, Type, TypeVar,
                    Union)

from pytorch_lightning import LightningDataModule
from utils.logging_utils import get_logger

from common import Metrics, Config
from .base.environment import Actions, Environment, Observations, Rewards
from .base.results import Results
from .method_abc import MethodABC

logger = get_logger(__file__)


class SettingABC(LightningDataModule):
    """ Abstract base class for a Setting.

    This just shows the minimal API. For more info, see the `Setting` class.

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
    def apply(self, method: MethodABC, config: Config) -> "SettingABC.Results":
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
        config : Config
            Configuration object with things like the log dir, the data dir,
            cuda, wandb config, etc.

        Returns
        -------
        Results
            An object that is used to measure or quantify the performance of the
            Method on this experimental Setting.
        """
        # For illustration purposes only:
        self.config = config
        method.config = config
        method.configure(self)
        method.fit(
            train_env=self.train_dataloader(),
            valid_env=self.val_dataloader(),
        )

        total_metrics = Metrics()
        test_environment = self.test_dataloader()
        for observations in test_environment:
            # Get the predictions/actions:
            actions = method.get_actions(observations, test_environment.action_space)
            # Get the rewards for the given predictions.
            rewards = test_environment.send(actions)
            # Calculate the 'metrics' (TODO: This should be done be in the env!)
            metrics = self.get_metrics(actions=actions, rewards=rewards)
            total_metrics += metrics
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

    ## Below this are some class attributes and methods related to the Tree.

    # These are some "private" class attributes.
    # For any new Setting subclass, it's parent setting.
    _parent: ClassVar[Type["Setting"]] = None
    # A list of all the direct children of this setting.
    _children: ClassVar[Set[Type["SettingABC"]]] = set()
    # List of all methods that directly target this Setting.
    _targeted_methods: ClassVar[Set[Type[MethodABC]]] = set()
    
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
    def get_applicable_methods(cls) -> List[Type[MethodABC]]:
        """ Returns all the Methods applicable on this Setting. """
        applicable_methods = []
        applicable_methods.extend(cls._targeted_methods)
        if cls._parent:
            applicable_methods.extend(cls._parent.get_applicable_methods())
        return applicable_methods

    @classmethod
    def register_method(cls, method: Type[MethodABC]):
        """ Register a method as being Applicable on this type of Setting. """
        cls._targeted_methods.add(method)

    @classmethod
    def get_name(cls) -> str:
        """ Gets the name of this Setting. """
        # LightningDataModule has a `name` class attribute of `...`!
        if getattr(cls, "name", None) != Ellipsis:
            return cls.name
        from utils import camel_case, remove_suffix
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
