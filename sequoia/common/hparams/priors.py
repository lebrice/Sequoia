import math
import random
from abc import abstractmethod
from dataclasses import InitVar, dataclass
from typing import TypeVar, Generic, Union

import numpy as np

T = TypeVar("T")


@dataclass  # type: ignore
class Prior(Generic[T]):
    rng: np.random.RandomState = np.random
    @abstractmethod
    def sample(self) -> T:
        pass
    
    @abstractmethod
    def get_orion_space_string(self) -> str:
        """ Gets the 'Orion-formatted space string' for this Prior object. """ 

@dataclass
class NormalPrior(Prior):
    mu: float = 0.
    sigma: float = 1.
    discrete: bool = False

    def sample(self) -> Union[float, int]:
        value = self.rng.normal(self.mu, self.sigma)
        if self.discrete:
            return round(value)
        return value

    def get_orion_space_string(self) -> str:
        raise NotImplementedError(
            "TODO: Add this for the normal prior, didn't check how its done in "
            "Orion yet."
        )

@dataclass
class UniformPrior(Prior):
    min: float = 0.
    max: float = 1.
    discrete: bool = False

    def sample(self) -> Union[float, int]:
        # TODO: add suport for enums?
        value = self.rng.uniform(self.min, self.max)
        if self.discrete:
            return round(value)
        return value

    def get_orion_space_string(self) -> str:
        string = f"uniform({self.min}, {self.max}"
        if self.discrete:
            string += ", discrete=True"
        string += ")"
        return string


@dataclass
class LogUniformPrior(Prior):
    min: float = 1e-3
    max: float = 1e+3
    base: float = np.e
    discrete: bool = False

    def sample(self) -> float:
        # TODO: Might not be 100% numerically stable.
        assert self.min > 0, "min of LogUniform can't be negative!"
        assert self.min < self.max, "max should be greater than min!"

        log_val = self.rng.uniform(self.log_min, self.log_max)
        value = math.pow(self.base, log_val)
        if self.discrete:
            return round(value)
        return value

    @property
    def log_min(self) -> Union[int, float]:
        if self.base is np.e:
            log_min = np.log(self.min)
        else:
            log_min = math.log(self.min, self.base)
        return log_min

    @property
    def log_max(self) -> Union[int, float]:
        if self.base is np.e:
            log_max = np.log(self.max)
        else:
            log_max = math.log(self.max, self.base)
        return log_max

    def get_orion_space_string(self) -> str:
        def format_power(value: float, log_value: float):
            if isinstance(value, int) or value.is_integer():
                return int(value)
            elif isinstance(log_value, int) or log_value.is_integer():
                log_value = int(log_value)
                if self.base == np.e:
                    return f"np.exp({int(log_value)})"
                elif self.base == 10:
                    return f"{value:.2e}"
            if math.log10(value).is_integer():
                return f"{value:.0e}"
            else:
                return f"{value:g}"
        
        min_str = format_power(self.min, self.log_min)
        max_str = format_power(self.max, self.log_max)
        string = f"loguniform({min_str}, {max_str}"
        if self.discrete:
            string += ", discrete=True"
        string += ")"
        return string



