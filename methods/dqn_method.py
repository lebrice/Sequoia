from .method import Method
from settings import Setting, RLSetting
from dataclasses import dataclass
from .models.dqn import DQN


@dataclass
class DQNMethod(Method, target_setting=RLSetting):
    """ Method aimed at solving an RL setting. """

    def model_class(self, setting: RLSetting):
        return DQN


if __name__ == "__main__":
    DQNMethod.main()