""" Utility script used to benchmark the speed of the BatchedVectorEnv,
depending on the environment, the batch size and the number of workers.  """
import time
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import gym

from common.gym_wrappers.batch_env import BatchedVectorEnv


def benchmark(env_fn: Union[str, Callable],
              batch_size: int,
              n_workers: int,
              wrappers: List[Callable]=None,
              n_steps: int = 100.,
              **kwargs):
    if isinstance(env_fn, str):
        env_fn = partial(gym.make, env_fn)
      
    start_time = time.time()
    env = BatchedVectorEnv([env_fn for i in range(batch_size)],
                           n_workers=n_workers, **kwargs)

    wrappers = wrappers or []
    for wrapper in wrappers:
        env = wrapper(env)
    
    setup_time = time.time() - start_time

    run_start = time.time()
    env.reset()
    with env:
        for i in range(n_steps):
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            env.render(mode="human")
            
    time_per_step = (time.time() - run_start) / n_steps
    return setup_time, time_per_step

def main():
    batch_size = 32
    n_steps = 100
    n_workers = None
    # from common.gym_wrappers.pixel_observation import PixelObservationWrapper
    env = "Breakout-v0"
    
    results: Dict[Tuple[int, int], float] = {}
    for batch_size in [1, 4, 8, 32, 64, 128]:
        for n_workers in [1, 2, 4, 8, None, batch_size]:
            if batch_size >= 32 and n_workers is not None:
                n_workers = max(n_workers, 4)

            setup_time, time_per_step = benchmark(
                env,
                batch_size,
                n_workers,
                n_steps=n_steps,
                context="fork",
            )
            observations_per_sec = round((1 / time_per_step) * batch_size)
            results[f"{batch_size}-{n_workers}"] = observations_per_sec
            print(f"batch size: {batch_size}, "
                  f"\tn_workers: {n_workers}, "
                  f"\tSetup time: {setup_time:.2f}, "
                  f"\tsteps/s: {1/time_per_step:.2f}, "
                  f"obs/s: {observations_per_sec:.1f}")
    import json
    print(json.dumps(results, indent="\t"))
    


if __name__ == "__main__":
    main()
