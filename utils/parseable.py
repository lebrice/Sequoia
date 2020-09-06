import shlex
import sys
from argparse import Namespace
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Tuple, Type, TypeVar, Union

from simple_parsing import ArgumentParser, ParsingError

from .logging_utils import get_logger

logger = get_logger(__file__)
T = TypeVar("T")

def from_args(cls: Type[T],
              argv: Union[str, List[str]] = None,
              reorder: bool = True,
              strict: bool = False) -> Tuple[T, Namespace]:
    logger.debug(f"parsing an instance of class {cls} from argv {argv}")
    if isinstance(argv, str):
        argv = shlex.split(argv)
    parser = ArgumentParser(description=cls.__doc__)
    dest = cls.__qualname__
    parser.add_arguments(cls, dest=dest)
    if not strict:
        args, unused_args = parser.parse_known_args(argv, attempt_to_reorder=reorder)
        if unused_args:
            logger.warning(UserWarning(
                f"Unknown/unused args when parsing class {cls}: {unused_args}"
            ))
        value: T = getattr(args, dest)
        return value, unused_args
    else:
        args = parser.parse_args(argv)
        value: T = getattr(args, dest)
        return value, Namespace()


@dataclass
class Parseable:
    _argv: Optional[List[str]] = field(default=None, init=False, repr=False)

    @classmethod
    def from_args(cls: Type[T],
                  argv: Union[str, List[str]] = None,
                  reorder: bool = True,
                  strict: bool = False) -> T:
        assert is_dataclass(cls), f"Can't get class {cls} from args, as it isn't a dataclass."
        if isinstance(argv, str):
            argv = shlex.split(argv)
        instance, _ = from_args(cls, argv=argv, reorder=reorder, strict=strict)
        # Save the argv that were used to create the instance on its `_argv`
        # attribute.
        instance._argv = argv or sys.argv
        return instance

    @classmethod
    def from_known_args(cls: Type[T],
                        argv: Union[str, List[str]] = None,
                        reorder=True) -> Tuple[T, Namespace]:
        assert is_dataclass(cls), f"Can't get class {cls} from args, as it isn't a dataclass."
        instance, unused_args = from_args(cls, argv=argv, reorder=reorder)
        instance._argv = argv or sys.argv
        return instance, unused_args
