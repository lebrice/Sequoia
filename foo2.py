from settings import ClassIncrementalSetting
from methods import RandomBaselineMethod

method = RandomBaselineMethod()
setting = ClassIncrementalSetting()
results = method.apply_to(setting)
print(f"Results: {results}")