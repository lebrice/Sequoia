""" Slightly modified version of SyncVectorEnv:

-   Doesn't manually reset the env during 'step' if it is a VectorEnv
-   Saves the final observation before a reset in the `info` dict at the
    FINAL_STATE_KEY key (current set to "final_state")

"""
import numpy as np
from copy import deepcopy
from gym.vector.vector_env import VectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv as SyncVectorEnv_
from gym.vector.sync_vector_env import concatenate

from .tile_images import tile_images
from .worker import FINAL_STATE_KEY


class SyncVectorEnv(SyncVectorEnv_):
    """ Subclassing the SyncVectorEnv from gym just so we can add in the changes
    from these open PRs:
    - https://github.com/openai/gym/pull/2072
    - https://github.com/openai/gym/pull/2104
    """
    
     
    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            # Don't manually reset VectorEnvs, since they reset the right env
            # themselves in `step`.
            if not isinstance(env, VectorEnv) and self._dones[i]:
                if info is None:
                    info = {}
                if FINAL_STATE_KEY not in info:
                    info[FINAL_STATE_KEY] = observation
                observation = env.reset()
            observations.append(observation)
            infos.append(info)
        concatenate(observations, self.observations, self.single_observation_space)

        return (deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards), np.copy(self._dones), infos)


    def render(self, mode: str = "rgb_array"):        
        image_batch = np.stack([env.render(mode="rgb_array") for env in self.envs])
        if mode == "rgb_array":
            return image_batch
        
        if mode == "human":
            tiled_version = tile_images(image_batch)
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(tiled_version)
            return self.viewer.isopen
        
        raise NotImplementedError(f"Unsupported mode {mode}")

    def close_extras(self, **kwargs):
        super().close_extras(**kwargs)
        if self.viewer:
            self.viewer.close()