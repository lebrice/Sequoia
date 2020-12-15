import glob
import inspect
import warnings
from importlib import import_module
from os.path import basename, dirname, isfile, join
from typing import List, Type

from sequoia.settings.base import Method

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
    if new_method not in all_methods:
        for method in all_methods:
            if method.get_name() == name:
                # BUG: There's this weird double-import thing happening during
                # testing, where some methods are import twice, first as
                # methods.baseline_method.BaselineMethod, for instance, then again
                # as SSCL.methods.baseline_method.BaselineMethod
                from os.path import abspath
                if abspath(inspect.getsourcefile(method)) == abspath(inspect.getsourcefile(new_method)):
                    # The two classes have the same name and are both defined in
                    # the same file, so this is basically the 'double-import bug
                    # described above.
                    break
                raise RuntimeError(f"There is already a registered method with name {name}: {method}")
        else:
            all_methods.append(new_method)
    return new_method


from .baseline_method import BaselineMethod
from .random_baseline import RandomBaselineMethod
from .experience_replay import ExperienceReplayMethod


## Pretty hacky: Dynamically import all the modules/packages defined in this
# folder. This way, we register the methods as they are declared.
modules = glob.glob(join(dirname(__file__), "*"))

all_modules: List[str] = [
    basename(f).replace(".py", "") for f in modules
    if (#(isfile(f) and
    not f.endswith('__init__.py') and
    not f.endswith("_test.py"))
]

for module in all_modules:
    try:
        import_module(f"sequoia.methods.{module}")
    except ImportError as e:
        warnings.warn(RuntimeWarning(f"Couldn't import Method from module methods/{module}: {e}"))

# TODO: (#17): Add Pl Bolts Models as Methods on IID Setting.
# from .pl_bolts_methods.cpcv2 import CPCV2Method

# print(" All methods: ", all_methods)
