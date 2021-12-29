# TODO: Need tests for putting fancier objects in the ReplayBuffer.
from sequoia.common.episode_collector.episode import Transition
from .replay_buffer import ReplayBuffer
from gym import spaces
import gym
from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors
from sequoia.common.typed_gym import _Space
from .off_policy_test import SimpleEnv



def test_with_typed_objects_and_tensors():
    # TODO: First, add tests for the env dataset / dataloader / experience replay with envs that
    # have typed objects (e.g.) Observation/Action/Reward, tensors, etc.
    env = SimpleEnv()
    from sequoia.methods.models.base_model.rl.base_model_rl import UseObjectsWrapper
    env = UseObjectsWrapper(env)
    env = ConvertToFromTensors(env, device="cpu")
    
    from .replay_buffer import ReplayBuffer
    from .episode_collector import EpisodeCollector
    
    
    def policy(obs: int, action_space: _Space[int]) -> int:
        return action_space.sample()

    max_episodes = 2
    loader = ReplayBuffer(item_Space=Transition.space_for_env(env), batch_size=3, policy=policy, max_episodes=max_episodes)
    i = 0
    for i, transitions in enumerate(loader):
        assert False, transitions
        # assert False, episode[len(episode)-1]

    assert i == max_episodes - 1