from io import StringIO
from contextlib import redirect_stdout
from gym import spaces
import numpy as np

# Avoid print statements from pygame package.
with redirect_stdout(StringIO()):
    from meta_monsterkong.make_env import MetaMonsterKongEnv
from sequoia.settings.rl.wrappers.state_vs_pixels import observe_state, observe_pixels


@observe_state.register
def observe_state_in_monsterkong(env: MetaMonsterKongEnv) -> MetaMonsterKongEnv:
    if not env.observe_state:
        env.unwrapped.observe_state = True
        env.unwrapped.observation_space = spaces.Box(0, 292, [402,], np.int16)
    return env


@observe_pixels.register
def observe_pixels_in_monsterkong(env: MetaMonsterKongEnv) -> MetaMonsterKongEnv:
    if env.observe_state:
        env.unwrapped.observe_state = False 
        env.unwrapped.observation_space = spaces.Box(0, 255, (64, 64, 3), np.uint8)
    return env
