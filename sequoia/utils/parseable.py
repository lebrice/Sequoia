import dataclasses
import shlex
import sys
from argparse import Namespace
from dataclasses import Field, dataclass, field, is_dataclass
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from pytorch_lightning import LightningDataModule
from simple_parsing import ArgumentParser

from sequoia.utils.utils import camel_case
from .logging_utils import get_logger

logger = get_logger(__file__)
P = TypeVar("P", bound="Parseable")


class Parseable:
    _argv: Optional[List[str]] = None

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = None) -> None:
        """Add the command-line arguments for this class to the given parser.
        
        Override this if you don't use simple-parsing to add the args.
        
        Parameters
        ----------
        parser : ArgumentParser
            The ArgumentParser. 
        dest : str, optional
            The 'base' destination where the arguments should be set on the
            namespace, by default None, in which case the arguments can be at
            the "root" level on the namespace.
        """
        if is_dataclass(cls):
            dest = dest or camel_case(cls.__qualname__)
            parser.add_arguments(cls, dest=dest)
        elif issubclass(cls, LightningDataModule):
            # TODO: Test this case out (using a LightningDataModule as a Setting).
            super().add_argparse_args(parser)  # type: ignore
        else:
            raise NotImplementedError(
                f"Don't know how to add command-line arguments for class "
                f"{cls}, since it isn't a dataclass and doesn't override the "
                f"`add_argparse_args` method!\n"
                f"Either make class {cls} a dataclass and add command-line "
                f"arguments as fields, or add an implementation for the "
                f"`add_argparse_args` and `from_argparse_args` classmethods."
            )

    @classmethod
    def from_argparse_args(cls: Type[P], args: Namespace, dest: str = None) -> P:
        """Extract the parsed command-line arguments from the namespace and
        return an instance of class `cls`.

        Override this if you don't use simple-parsing.

        Parameters
        ----------
        args : Namespace
            The namespace containing all the parsed command-line arguments.
        dest : str, optional
            The , by default None

        Returns
        -------
        cls
            An instance of the class `cls`.
        """
        if is_dataclass(cls):
            dest = dest or camel_case(cls.__qualname__)
            return getattr(args, dest)

        # if issubclass(cls, LightningDataModule):
        #     # TODO: Test this case out (using a LightningDataModule as a Setting).
        #     return super()._from_argparse_args(args)  # type: ignore

        raise NotImplementedError(
            f"Don't know how to extract the command-line arguments for class "
            f"{cls} from the namespace, since {cls} isn't a dataclass and "
            f"doesn't override the `from_argparse_args` classmethod."
        )

    @classmethod
    def from_args(cls: Type[P],
                  argv: Union[str, List[str]] = None,
                  reorder: bool = True,
                  strict: bool = True) -> P:
        """Parse an instance of this class from the command-line args.

        Parameters
        ----------
        cls : Type[P]
            The class to instantiate. This only supports dataclasses by default.
            For other classes, you'll have to implement this method yourself.
        argv : Union[str, List[str]], optional
            The command-line string or list of string arguments in the style of
            sys.argv. Could also be the unused_args returned by
            .from_known_args(), for example. By default None
        reorder : bool, optional
            Wether to attempt to re-order positional arguments. Only really
            useful when using subparser actions. By default True.
        strict : bool, optional
            Wether to raise an error if there are extra arguments. By default
            False

            TODO: Might be a good idea to actually change this default to 'True'
            to avoid potential subtle bugs in various places. This would however
            make the code slightly more difficult to read, since we'd have to
            pass some unused_args around. Also might be a problem when the same
            argument e.g. batch_size (at some point) is in both the Setting and
            the Method, because then the arg would be 'consumed', and not passed
            to the second parser in the chain.

        Returns
        -------
        P
            The parsed instance of this class.

        Raises
        ------
        NotImplementedError
            [description]
        """
        # if not is_dataclass(cls):
        #     raise NotImplementedError(
        #         f"Don't know how to create an instance of class {cls} from the "
        #         f"command-line, as it isn't a dataclass. You'll have to "
        #         f"override the `from_args` or `from_known_args` classmethods."
        #     )
        if isinstance(argv, str):
            argv = shlex.split(argv)
        instance, unused_args = cls.from_known_args(
            argv=argv,
            reorder=reorder,
            strict=strict,
        )
        assert not (strict and unused_args), "an error should have been raised"
        return instance

    @classmethod
    def from_known_args(cls,
                        argv: Union[str, List[str]] = None,
                        reorder: bool = True,
                        strict: bool = False) -> Tuple[P, List[str]]:
        # if not is_dataclass(cls):
        #     raise NotImplementedError(
        #         f"Don't know how to parse an instance of class {cls} from the "
        #         f"command-line, as it isn't a dataclass or doesn't have the "
        #         f"`add_arpargse_args` and `from_argparse_args` classmethods. "
        #         f"You'll have to override the `from_known_args` classmethod."
        #     )

        if argv is None:
            argv = sys.argv[1:]
        logger.debug(f"parsing an instance of class {cls} from argv {argv}")
        if isinstance(argv, str):
            argv = shlex.split(argv)

        parser = ArgumentParser(description=cls.__doc__,
                                add_dest_to_option_strings=False)
        cls.add_argparse_args(parser)
        # TODO: Set temporarily on the class, so its accessible in the class constructor
        cls_argv = cls._argv
        cls._argv = argv

        instance: P
        if strict:
            args = parser.parse_args(argv)
            unused_args = []
        else:
            args, unused_args = parser.parse_known_args(argv, attempt_to_reorder=reorder)
            if unused_args:
                logger.debug(RuntimeWarning(
                    f"Unknown/unused args when parsing class {cls}: {unused_args}"
                ))
        instance = cls.from_argparse_args(args)
        # Save the argv that were used to create the instance on its `_argv`
        # attribute.
        instance._argv = argv
        cls._argv = cls_argv
        return instance, unused_args

    def upgrade(self, target_type: Type[P]) -> P:
        """Upgrades the hparams `self` to the given `target_type`, filling in
        any missing values by parsing them from the command-line.

        If `self` was created from the command-line, then the same argv that
        were used to create `self` will be used to create the new object.

        Returns
        -------
        type(self).HParams
            Hparams of the type `self.HParams`, with the original values
            preserved and any new values parsed from the command-line.
        """
        # NOTE: This (getting the wrong hparams class) could happen for
        # instance when parsing a BaseMethod from the command-line, the
        # default type of hparams on the method is BaseModel.HParams,
        # whose `output_head` field doesn't have the right type exactly.
        current_type = type(self)
        current_hparams = dataclasses.asdict(self)
        # NOTE: If a value is not at its current default, keep it.
        default_hparams = target_type()
        missing_fields = [
            f.name for f in dataclasses.fields(target_type)
            if f.name not in current_hparams
            or current_hparams[f.name] == getattr(current_type(), f.name, None)
            or current_hparams[f.name] == getattr(default_hparams, f.name)
        ]
        logger.warning(RuntimeWarning(
            f"Upgrading the hparams from type {current_type} to "
            f"type {target_type}. This will try to fetch the values for "
            f"the missing fields {missing_fields} from the command-line. "
        ))
        # Get the missing values

        if self._argv:
            return target_type.from_args(argv=self._argv, strict=False)
        hparams = target_type.from_args(argv=self._argv, strict=False)
        for missing_field in missing_fields:
            current_hparams[missing_field] = getattr(hparams, missing_field)
        return target_type(**current_hparams)


    # @classmethod
    # def fields(cls) -> Dict[str, Field]:
    #     return {f.name: f for f in dataclasses.fields(cls)}
