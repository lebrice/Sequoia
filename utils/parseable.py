import shlex
from argparse import Namespace
from dataclasses import dataclass, is_dataclass
from typing import List, Tuple, Type, TypeVar, Union

from simple_parsing import ArgumentParser

from .logging_utils import get_logger

logger = get_logger(__file__)
T = TypeVar("T")

def from_args(cls: Type[T], argv: Union[str, List[str]]=None, reorder: bool=True) -> Tuple[T, Namespace]:
    logger.debug(f"parsing an instance of class {cls} from argv {argv}")
    if isinstance(argv, str):
        argv = shlex.split(argv)
    parser = ArgumentParser(description=cls.__doc__)
    dest = cls.__qualname__
    parser.add_arguments(cls, dest=dest)
    args, unused_args = parser.parse_known_args(argv, attempt_to_reorder=reorder)
    if unused_args:
        logger.warning(UserWarning(
            f"Unknown/unused args when parsing class {cls}: {unused_args}"
        ))
    value: T = getattr(args, dest)
    return value, unused_args


@dataclass
class Parseable:
    @classmethod
    def from_args(cls: Type[T],
                  argv: Union[str, List[str]] = None,
                  reorder=True) -> T:
        assert is_dataclass(cls), f"Can't get class {cls} from args, as it isn't a dataclass."
        instance, _ = from_args(cls, argv=argv, reorder=reorder)
        return instance

    @classmethod
    def from_known_args(cls: Type[T],
                        argv: Union[str, List[str]] = None,
                        reorder=True) -> Tuple[T, Namespace]:
        assert is_dataclass(cls), f"Can't get class {cls} from args, as it isn't a dataclass."
        return from_args(cls, argv=argv, reorder=reorder)
