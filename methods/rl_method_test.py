from .method import Method
from settings import Setting, RLSetting

from .rl_method import RLMethod

def test_basics():
    method = RLMethod.from_args("--debug --fast_dev_run")
    setting = RLSetting(observe_state_directly = True)
    results = method.apply_to(setting)
    assert results is not None
