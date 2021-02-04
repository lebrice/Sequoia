from sequoia.settings import TaskIncrementalRLSetting
from sequoia.common import Config
from sequoia.methods import RandomBaselineMethod

if __name__ == "__main__":
    setting = TaskIncrementalRLSetting(dataset="cartpole", observe_state_directly=True, nb_tasks=2)
    method = RandomBaselineMethod(batch_size=1)
    results = setting.apply(method, Config(num_workers=0, debug=True))
    print(f"results: {results.summary()}")
