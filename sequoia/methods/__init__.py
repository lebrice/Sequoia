import glob
import inspect
import os
import warnings
from importlib import import_module
from os.path import basename, dirname, isfile, join
from pathlib import Path
from typing import List, Type

from setuptools import find_packages

from sequoia.settings.base import Method
from sequoia.utils.logging_utils import get_logger
logger = get_logger(__file__)


AbstractMethod = Method

all_methods: List[Type[Method]] = []

"""
TODO: IDEA: Add arguments to register_method that help configure the tests we
add the that method! E.g.:

```
@register_method(slow=True, requires_cuda=True, required_memory_gb=4)
class MyMethod(Method, target_setting=ContinualRLSetting):
    ...
```
"""


def register_method(new_method: Type[Method]) -> Type[Method]:
    name = new_method.get_name()
    # print(f"Registering method with name {name}")
    if new_method not in all_methods:
        for method in all_methods:
            if method.get_name() == name:
                # BUG: There's this weird double-import thing happening during
                # testing, where some methods are import twice, first as
                # methods.baseline_method.BaselineMethod, for instance, then again
                # as SSCL.methods.baseline_method.BaselineMethod
                from os.path import abspath
                method_source_file = inspect.getsourcefile(method)
                assert isinstance(method_source_file, str), f"cant find source file of {method}?"
                new_method_source_file = inspect.getsourcefile(new_method)
                assert isinstance(new_method_source_file, str), f"cant find source file of {new_method}?"

                if abspath(method_source_file) == abspath(new_method_source_file):
                    # The two classes have the same name and are both defined in
                    # the same file, so this is basically the 'double-import bug
                    # described above.
                    break

                method_family = method.get_family()
                new_method_family = new_method.get_family()
                assert method_family != new_method_family, (
                    "Can't have two methods with the same name in the same family!"
                )
        else:
            all_methods.append(new_method)
    return new_method

# NOTE: Even though these methods would be dynamically registered (see below),
# we still import them so we can do `from methods import BaselineMethod`.
from .baseline_method import BaselineMethod
from .random_baseline import RandomBaselineMethod
from .pnn import PnnMethod


try:
    from .avalanche import *
except ImportError:
    pass


try:
    from .stable_baselines3_methods import *
except ImportError:
    pass


try:
    from .pl_bolts_methods import *
except ImportError:
    pass

try:
    # TODO: This won't work, because we don't currently package the `cndpm_method`
    # inside of the `cndpm` package.
    # from cn_dpm.cndpm_method import CNDPM

    # NOTE However: This will following line WILL work, because we point to the actual
    # file to use. This is tricky though: If you just do `pip install -e .[cn_dpm],
    # then the `cn_dpm` package gets copied from `sequoia/methods/cn_dpm` into
    # site_packages, and then using `from cn_dpm.models import ...` actually imports it
    # from the COPY, instead of from the actual submodule!
    # TODO: (@lebrice): figure out a way to install cn-dpm correctly through pip, in
    # editable mode.
    # For now though, you should install the CN-DPM submodule in editable mode, like so:
    # `pip install -e sequoia/methods/cn_dpm`
    from .cn_dpm.cndpm_method import CNDPM

except ImportError:
    pass

## A bit hacky: Dynamically import all the modules/packages defined in this
# folder. This way, we register the methods as they are declared.
source_dir = Path(os.path.dirname(__file__))

all_modules = find_packages(where=source_dir)
# Add all non-package modules (i.e. all *.py files in this methods folder), for example
# ewc_method.py.
all_modules.extend(
    [
        file.relative_to(source_dir).stem for file in Path(source_dir).glob("*.py")
        if not file.name.endswith("_test.py") and file.name != "__init__.py"
    ]
)

for module in all_modules:
    try:
        # print(f"Importing module sequoia.methods.{module}")
        import_module(f"sequoia.methods.{module}")
    except ImportError as e:
        logger.warning(RuntimeWarning(f"Couldn't import Method from module methods/{module}: {e}"))

# TODO: (#17): Add Pl Bolts Models as Methods on IID Setting.
# from .pl_bolts_methods.cpcv2 import CPCV2Method
