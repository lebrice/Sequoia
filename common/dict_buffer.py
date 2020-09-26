from collections import OrderedDict, abc
from typing import Iterable, Iterator, Mapping, Optional, Tuple

from torch import Tensor
from torch.nn.modules import Module, ParameterDict


class DictBuffer(ParameterDict):
    def __init__(self, parameters: Optional[Mapping[str, Tensor]] = None) -> None:
        super().__init__()
        # Since we can't register a buffer with a "." in it, we 
        dot = "."
        dot_ = "^"
        self.translate = lambda k:  k.replace(dot, dot_)
        self.untranslate =  lambda k: k.replace(dot_, dot)
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key: str) -> 'Parameter':
        key = self.translate(key)
        return self._parameters[key]

    def __setitem__(self, key: str, parameter: 'Parameter') -> None:
        key = self.translate(key)
        self.register_buffer(key, parameter)

    def __delitem__(self, key: str) -> None:
        key = self.translate(key)
        del self._parameters[key]

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(map(self.untranslate, self._parameters.keys()))

    def items(self) -> Iterable[Tuple[str, Tensor]]:
        for k, v in self._parameters.items():
            yield self.untranslate(k), v

    def __contains__(self, key: str) -> bool:
        key = self.translate(key)
        return super().__contains__(key)

    def pop(self, key: str) -> 'Parameter':
        r"""Remove key from the ParameterDict and return its parameter.

        Arguments:
            key (string): key to pop from the ParameterDict
        """
        key = self.translate(key)
        return super().pop(key)

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys.
        """
        return list(map(self.untranslate, self._parameters.keys()))

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            k = self.untranslate(k)
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr
    
    def update(self, parameters: Mapping[str, 'Parameter']) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        for k, v in parameters.items():
            self[k] = v
        return
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError("ParameterDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(p).__name__)
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(p)) +
                                     "; 2 is required")
                self[p[0]] = p[1]


    def __call__(self, input):
        raise RuntimeError('ParameterDict should not be called.')
