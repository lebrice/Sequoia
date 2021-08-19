from sequoia.settings.sl import TaskIncrementalSLSetting
from sequoia.methods.packnet_method import PackNetMethod


if __name__ == '__main__':
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        nb_tasks=2
    )

    my_method = PackNetMethod()
    results = setting.apply(my_method)