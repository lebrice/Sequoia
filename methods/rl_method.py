from .method import Method
from settings import Setting, RLSetting
from dataclasses import dataclass
from .models.agent import Agent
from typing import Type


@dataclass
class RLMethod(Method, target_setting=RLSetting):
    """ Method aimed at solving an RL setting. """

    def model_class(self, setting: RLSetting) -> Type[Agent]:
        return Agent


if __name__ == "__main__":
    RLMethod.main()