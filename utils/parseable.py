import shlex
import sys
from abc import abstractmethod, ABC
from argparse import Namespace
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Tuple, Type, TypeVar, Union

from simple_parsing import ArgumentParser, ParsingError

from .logging_utils import get_logger

logger = get_logger(__file__)
P = TypeVar("T", bound="Parseable")

def from_args(cls: Type[P],
              argv: Union[str, List[str]] = None,
              reorder: bool = True,
              strict: bool = False) -> Tuple[P, Namespace]:
    if argv is None:
        argv = sys.argv[1:]
    logger.debug(f"parsing an instance of class {cls} from argv {argv}")
    if isinstance(argv, str):
        argv = shlex.split(argv)
    parser = ArgumentParser(description=cls.__doc__)
    
    cls.add_argparse_args(parser)
    
    instance: P
    if not strict:
        args, unused_args = parser.parse_known_args(argv, attempt_to_reorder=reorder)
        if unused_args:
            logger.warning(UserWarning(
                f"Unknown/unused args when parsing class {cls}: {unused_args}"
            ))
    else:
        args = parser.parse_args(argv)
        unused_args = Namespace()
    
    instance = cls.from_argparse_args(args)
    return instance, unused_args

@dataclass
class Parseable:
    _argv: Optional[List[str]] = field(default=None, init=False, repr=False)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> None:
        """ Adds the command-line arguments for this class to the parser.
        
        Override this if you don't use simple-parsing to add the args.
        """
        dest = cls.__qualname__
        parser.add_arguments(cls, dest=dest)
        
    @classmethod
    def from_argparse_args(cls: Type[P], args: Namespace) -> P:
        """ Creates an instance of this class from the parsed arguments.
        
        Override this if you don't use simple-parsing.
        """
        dest = cls.__qualname__
        return getattr(args, dest)
    
    @classmethod
    def from_args(cls: Type[P],
                  argv: Union[str, List[str]] = None,
                  reorder: bool = True,
                  strict: bool = False) -> P:
        assert is_dataclass(cls), f"Can't get class {cls} from args, as it isn't a dataclass."
        if isinstance(argv, str):
            argv = shlex.split(argv)
        instance, _ = from_args(cls, argv=argv, reorder=reorder, strict=strict)
        # Save the argv that were used to create the instance on its `_argv`
        # attribute.
        instance._argv = argv or sys.argv
        return instance

    @classmethod
    def from_known_args(cls: Type[P],
                        argv: Union[str, List[str]] = None,
                        reorder=True) -> Tuple[P, Namespace]:
        assert is_dataclass(cls), f"Can't get class {cls} from args, as it isn't a dataclass."
        instance, unused_args = from_args(cls, argv=argv, reorder=reorder)
        instance._argv = argv or sys.argv
        return instance, unused_args
