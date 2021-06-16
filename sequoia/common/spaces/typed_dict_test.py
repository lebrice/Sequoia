import gym
import numpy as np
from gym import Space, spaces
from gym.spaces import Box, Discrete
from gym.vector.utils import batch_space
from typing import Tuple
from .typed_dict import TypedDictSpace


def test_basic():
    space = TypedDictSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
    )
    v = space.sample()
    print(v)
    assert v in space
    # TODO: Maybe re-use all the tests for gym.spaces.Tuple in the gym repo
    # somehow?

    vanilla_space = spaces.Dict(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
    )
    assert vanilla_space.sample() in space
    assert space.sample() in vanilla_space


def test_supports_dataclasses():
    # IDEA: Wrapper that makes the 'default factory' of each field actually use
    # the 'sample' method from a space associated with each class.

    @dataclass
    class Sample:
        a: np.ndarray
        b: bool
        c: Tuple[int, int]

    space = spaces.Dict(
        a=spaces.Box(0, 1, [2, 2]),
        b=spaces.Box(False, True, (), np.bool),
        c=spaces.MultiDiscrete([2, 2]),
    )

    wrapped_space: TypedDictSpace = TypedDictSpace(spaces=space.spaces, dtype=Sample)
    assert isinstance(wrapped_space, spaces.Dict)
    assert (
        Sample(a=np.ones([2, 2]), b=np.array(False), c=np.array([0, 1]),)
        in wrapped_space
    )
    assert isinstance(wrapped_space.sample(), Sample)


try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from dataclasses import dataclass, fields, Field
from typing import Mapping, Union, TypeVar, Dict, Iterable

T = TypeVar("T")


@dataclass
class StateTransition(Mapping[str, T]):
    current_state: T
    action: int
    next_state: T

    def __post_init__(self):
        self._fields: Dict[str, Field] = {f.name: f for f in fields(self)}

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, attr: str) -> T:
        if attr not in self._fields:
            raise KeyError(attr)
        return getattr(self, attr)

    def __iter__(self) -> Iterable[str]:
        return iter(self._fields)


def test_basic_with_dtype():
    space = TypedDictSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    v = space.sample()
    assert v in space
    assert isinstance(v, StateTransition)

    normal_space = spaces.Dict(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
    )
    assert normal_space.sample() in space
    # NOTE: this doesn't work when using a dtype that isn't a subclass of dict!
    if issubclass(space.dtype, dict):
        assert space.sample() in normal_space


def test_isinstance():
    space = TypedDictSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    assert isinstance(space, spaces.Dict)
    assert isinstance(space.sample(), StateTransition)


def test_equals_dict_space_with_same_items():
    """ Test that a TypedDictSpace is considered equal to aDict space if
    the spaces are in the same order and all equal.
    """
    space = TypedDictSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    dict_space = spaces.Dict(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
    )
    assert space == dict_space
    assert dict_space == space


def test_batch_objets_considered_valid_samples():
    from dataclasses import dataclass, field

    import numpy as np
    from sequoia.common.batch import Batch

    @dataclass(frozen=True)
    class StateTransitionDataclass(Batch):
        current_state: np.ndarray
        action: int
        next_state: np.ndarray

    space = TypedDictSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransitionDataclass,
    )
    obs = StateTransitionDataclass(
        current_state=np.ones([2, 2]) / 2, action=1, next_state=np.zeros([2, 2]),
    )
    assert obs in space
    assert space.sample() in space
    assert isinstance(space.sample(), StateTransitionDataclass)


def test_batch_space():
    space = TypedDictSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    assert batch_space(space, n=5) == TypedDictSpace(
        current_state=Box(0, 1, (5, 2, 2)),
        action=spaces.MultiDiscrete([2, 2, 2, 2, 2]),
        next_state=Box(0, 1, (5, 2, 2)),
        dtype=StateTransition,
    )


def test_batch_space_preserves_dtype():
    space = TypedDictSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    batched_space = batch_space(space, n=5)
    assert isinstance(batched_space, TypedDictSpace)
    assert list(batched_space.spaces.keys()) == list(batched_space.spaces.keys())
    assert list(batched_space.spaces.keys()) == [
        "current_state",
        "action",
        "next_state",
    ]
    assert batched_space.dtype is StateTransition

    space = TypedDictSpace(
        dict(
            current_state=Box(0, 1, (2, 2)),
            action=Discrete(2),
            next_state=Box(0, 1, (2, 2)),
        ),
        dtype=StateTransition,
    )
    batched_space = batch_space(space, n=5)
    assert isinstance(batched_space, TypedDictSpace)
    assert list(batched_space.spaces.keys()) == list(batched_space.spaces.keys())
    assert list(batched_space.spaces.keys()) == [
        "current_state",
        "action",
        "next_state",
    ]
    assert list(batched_space.sample().keys()) == [
        "current_state",
        "action",
        "next_state",
    ]
    assert list(v[0] for v in space.spaces.items()) == [
        "current_state",
        "action",
        "next_state",
    ]
    assert batched_space.dtype is StateTransition

    space = TypedDictSpace(
        dict(x=Box(0, 1, (2, 2)), action=Discrete(2), next_state=Box(0, 1, (2, 2)),),
    )
    batched_space = batch_space(space, n=5)
    assert batched_space.x == Box(0, 1, (5, 2, 2))
    assert isinstance(batched_space, TypedDictSpace)
    assert list(batched_space.spaces.keys()) == list(batched_space.spaces.keys())
    assert list(batched_space.spaces.keys()) == ["x", "action", "next_state"]
    assert list(batched_space.sample().keys()) == ["x", "action", "next_state"]
    assert list(v[0] for v in space.spaces.items()) == ["x", "action", "next_state"]


class DummyDictEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = TypedDictSpace(
            x=Box(0, 1, (2, 2)), t=Discrete(2), done=Box(False, True, (1,), bool),
        )
        self.action_space = spaces.Discrete(10)
        self.reward_space = spaces.Box(-10, 10, shape=(1,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()
    
    def step(self, action):
        return self.observation_space.sample(), self.reward_space.sample(), False, {}

    def seed(self, seed=None):
        seeds = []
        seeds += self.observation_space.seed(seed)
        seeds += self.action_space.seed(seed)
        seeds += self.reward_space.seed(seed)
        return seeds


def test_vector_env():
    env = DummyDictEnv()
    from gym.vector import make
    from gym.envs.registration import registry, register, EnvSpec
    register("dummy_foo-v0", entry_point=DummyDictEnv)
    env = make("dummy_foo-v0", num_envs=10)


from sequoia.common.batch import Batch
from torch import Tensor
from typing import Optional
from numpy.typing import ArrayLike


def test_object_with_extra_keys_fits():
    @dataclass(frozen=True)
    class Observation(Batch):
        x: np.ndarray
        t: ArrayLike
        done: Optional[ArrayLike] = None

    space = TypedDictSpace(
        x=spaces.Box(0, 10, (10,)), t=spaces.Box(0, 1, (1,), dtype=np.int32)
    )

    obs = Observation(x=np.arange(10), t=np.array([1]), done=False,)
    assert obs.x in space.x
    assert obs.t in space.t
    assert obs in space



def test_order_of_keys_is_same_in_samples():
    space = TypedDictSpace(
        x=spaces.Box(0, 10, (10,), dtype=np.int32), t=spaces.Discrete(10)
    )
    expected = ["x", "t"]
    assert list(space.keys()) == expected
    assert list(k for k, v in space.items()) == expected
    
    assert list(space.sample().keys()) == expected
    assert list(k for k, v in space.sample().items()) == expected
    space.seed(123)
    s = space.sample()
    assert str(s) == f"{{'x': {repr(s['x'])}, 't': {repr(s['t'])}}}"


def test_debugging():
    assert {'task_labels': 0, 'x': np.array([-0.25162117, -0.43992427,  0.42706016,  1.47862901])} in TypedDictSpace(x=spaces.Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), np.float32), task_labels=spaces.Discrete(5), dtype=dict)


## IDEA: Creating a space like this, using the same syntax as with TypedDict
# class StateTransitionSpace(TypedDict):
#     current_state: Box = Box(0, 1, (2,2))
#     action: Discrete = Discrete(2)
#     current_state: Box = Box(0, 1, (2,2))

# space = StateTransitionSpace()
# space.sample()
