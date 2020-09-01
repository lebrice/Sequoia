import inspect
import shlex
from abc import abstractmethod
from argparse import Namespace
from collections import OrderedDict
from dataclasses import InitVar, dataclass, fields, is_dataclass
from typing import *

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from common.config import Config
from common.loss import Loss
from common.transforms import Compose, Transforms
from simple_parsing import (ArgumentParser, Serializable, list_field,
                            mutable_field, subparsers)
from utils import Parseable, camel_case, dict_union, get_logger, remove_suffix

from .results import Results

logger = get_logger(__file__)

Loader = TypeVar("Loader", bound=DataLoader)
ResultsType = TypeVar("ResultsType", bound=Results)
SettingType = TypeVar("SettingType", bound="Setting")

from pytorch_lightning.core.datamodule import _DataModuleWrapper


class SettingWrapper(_DataModuleWrapper):
    """ TODO: We might want to move the command-line arguments from the Setting
    to a 'Config' class instead? There are issues with using a dataclass over a
    LightningDataModule atm (because LightningDataModule has __init__)
    """
    def __call__(cls, *args, **kwargs):
        """A wrapper for LightningDataModule that:

            1. Runs user defined subclass's __init__
            2. Assures prepare_data() runs on rank 0
            3. Lets you check prepare_data and setup to see if they've been called
        """
        init_fields: List[str] = [f.name for f in fields(cls) if f.init]
        extra_args: Dict[str, Any] = {}
        for k in list(kwargs.keys()):
            if k not in init_fields:
                extra_args[k] = kwargs.pop(k)
        if extra_args:
            logger.warning(UserWarning(
                f"Ignoring args {extra_args} when creating class {cls}."
            ))
        return super().__call__(*args, **kwargs)


@dataclass(init=True)
class Setting(LightningDataModule, Serializable, Parseable, Generic[Loader], metaclass=SettingWrapper):
    """Extends LightningDataModule to allow setting the transforms and options
    from the command-line.

    This class is Generic, which allows us to pass a different `Loader` type, 
    which should be the type of dataloader returned by the `train_dataloader`,
    `val_dataloader` and `test_dataloader` methods.
    """
    # Overwrite this in a subclass to customize which type of Results to create.
    results_class: ClassVar[Type[Results]] = Results

    _subclasses: ClassVar[List[Type["Setting"]]] = []

    _methods: ClassVar[Set[Type]] = set()
    # Transforms to be used.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels)

    # TODO: Currently trying to find a way to specify the transforms from the command-line.
    # As a consequence, don't try to set these from the command-line for now.
    train_transforms: List[Transforms] = list_field()

    # TODO: These two aren't being used atm (at least not in ClassIncremental), 
    # since the CL Loader from Continuum only takes in common_transforms and
    # train_transforms.
    val_transforms: List[Transforms] = list_field()
    test_transforms: List[Transforms] = list_field()

    # fraction of training data to devote to validation. Defaults to 0.2.
    val_fraction: float = 0.2

    # obs_shape: InitVar[Tuple[int, ...]] = ()
    # action_shape: InitVar[Tuple[int, ...]] = ()
    # reward_shape: InitVar[Tuple[int, ...]] = ()
    

    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):
        """ Initializes the fields of the setting that weren't set from the
        command-line.
        """
        train_transforms: Callable = Compose(self.train_transforms or self.transforms)
        val_transforms: Callable = Compose(self.val_transforms or self.transforms)
        test_transforms: Callable = Compose(self.test_transforms or self.transforms)

        self._dims: Tuple[int, ...] = ()
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )
        # TODO: Should we ask every setting to set these three properties ?
        self.obs_dims: Tuple[int, ...] = obs_shape
        self.action_dims: Tuple[int, ...] = action_shape
        self.reward_dims: Tuple[int, ...] = reward_shape
        self.dataloader_kwargs: Dict[str, Any] = {}
        # Wether the setup methods have been called yet or not.
        self._setup: bool = False
        self._prepared: bool = False
        self._configured: bool = False
        self.config: Config = None

    def apply(self, method: "Method") -> Results:
        """ Applies a method on this experimental setting.
        
        Extend this class and overwrite this method to customize your
        training/evaluation protocol.
        """
        # 1. Configure the method to work on the setting.
        method.configure(self)
        # 2. Train the method on the setting.
        method.train(self)
        # 3. Evaluate the method on the setting and return the results.
        return self.evaluate(method)

    def evaluate(self, method: "Method") -> ResultsType:
        """Tests the method and returns the Results.

        Overwrite this to customize testing for your experimental setting.

        Returns:
            ResultsType: A Results object for this particular setting.
        """
        from methods import Method
        method: Method
        trainer = method.trainer

        # Run the actual evaluation.
        test_outputs = method.trainer.test(
            datamodule=self,
            verbose=False,
        )
        test_loss: Loss = test_outputs[0]["loss_info"]

        model = method.model
        from methods.models import Model
        if isinstance(model, Model):
            hparams = model.hp
        else:
            assert False, f"TODO: Remove this ({model})."
            hparams = model.hparams
        return self.results_class(
            hparams=hparams,
            test_loss=test_loss,
        )

    @classmethod
    def main(cls, argv: Optional[Union[str, List[str]]]=None) -> ResultsType:
        from main import Experiment
        experiment: Experiment
        # Create the Setting object from the command-line:
        setting = cls.from_args(argv)
        # Then create the 'Experiment' from the command-line, which makes it
        # possible to choose between all the methods.
        experiment = Experiment.from_args(argv)
        # fix the setting attribute to be the one parsed above.
        experiment.setting = setting
        results: ResultsType = experiment.launch(argv)
        return results

    def apply_all(self, argv: Union[str, List[str]]=None) -> Dict[Type["Method"], Results]:
        applicable_methods = self.get_all_applicable_methods()

        all_results: Dict[Type[Method], Results] = OrderedDict()
        for method_type in applicable_methods:
            method = method_type.from_args(argv)
            results = method.apply(self)
            all_results[method_type] = results
        logger.info(f"All results for setting of type {type(self)}:")
        logger.info({
            method.get_name(): (results.get_metric() if results else "crashed")
            for method, results in all_results.items()
        })
        return all_results

    def configure(self, config: Config, **dataloader_kwargs):
        """ Set some of the misc options in the setting which might come from
        the Method or the Experiment.
        
        TODO: This isn't super clean, but we basically want to be able to give
        the batch_size, data_dir, num_workers etc to the Setting somehow,
        without letting it "know" what kind of method is being applied to it.
        """
        self.config = config
        self.data_dir = config.data_dir
        self.dataloader_kwargs.update(num_workers=config.num_workers)
        self.dataloader_kwargs.update(dataloader_kwargs)
        self._configured = True

    def setup_if_needed(self, *args, **kwargs) -> bool:
        if not self._configured:
            # set the data_dir and the dataloader kwargs
            self.configure(config=self.config or Config())
            self._configured = True
        # TODO: Don't know how to properly check when this is the 'main worker'.
        # (We should only download data on the main worker).
        if not self._prepared and (self.config.device.type == "cpu" or
                                   torch.cuda.device_count() == 1):
            self.prepare_data()
            self._prepared = True
        if not self._setup:
            self.setup(*args, **kwargs)
            self._setup = True

    @property
    @abstractmethod
    def dims(self) -> Tuple[int, ...]:
        """The input shape for this setting.

        Returns:
            Tuple[int, ...]: The input shape for this setting.
        """
        return self._dims

    @dims.setter
    def dims(self, value) -> None:
        self._dims = value
    

    @classmethod
    def is_applicable(cls: Type["Setting"], method: Union["Method", Type["Method"]]) -> bool:
        """Returns wether a Method is applicable to this setting type.

        A method is applicable to any setting which is an instance (or subclass)
        of the setting the method was defined to support initially.

        Args:
            method (Union[Method, Type[Method]]): a Method or method type.

        Returns:
            bool: Wether or not the given method is applicable to this setting.
        """
        from methods import Method
        if isinstance(method, Method):
            method_type = type(method)
        else:
            method_type = method
        return method_type.is_applicable(cls)

    @classmethod
    def get_all_applicable_methods(cls) -> List[Type["Method"]]:
        from methods import all_methods, Method
        return list(filter(cls.is_applicable, all_methods))

    def __init_subclass__(cls, **kwargs):
        assert is_dataclass(cls), f"Setting type {cls} isn't a dataclass!"
        logger.debug(f"Registering a new setting! {cls.get_name()}")
        # cls._subclasses.append(cls)
        # for t in cls.mro()[1:]:
        #     if inspect.isclass(t) and issubclass(t, Setting):
        #         t._subclasses.append(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_name(cls) -> str:
        """ Gets the name of this setting class. """
        # LightningDataModule has a `name` class attribute of `...`!
        if getattr(cls, "name", None) != Ellipsis:
            return cls.name
        else:
            name = camel_case(cls.__qualname__)
            return remove_suffix(name, "_setting")

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        if not is_dataclass(cls):
            return super().add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False,)
        dest: str = cls.__qualname__
        parser.add_arguments(cls, dest=dest)
        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        dest: str = cls.__qualname__
        if hasattr(args, dest):
            instance = args.dest
            assert not kwargs, f"kwargs: {kwargs}"
            return instance
        return super().from_argparse_args(args=args, **kwargs)
