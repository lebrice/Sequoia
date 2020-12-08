import numpy as np
import tqdm as tqdm
from typing import List


def get_fraction_of_observations_with_grad(n_envs: int,
                                           n_updates: int = 10,
                                           min_episode_length: int = 1,
                                           max_episode_length: int = 100):
    n_used_steps = 0
    n_wasted_steps = 0

    # min_episode_length = 0
    # max_episode_length = 10

    # The starting episode lengths for each env.
    # new_episode_length = lambda: 10
    # episode_lengths = [5, 10]
    
    new_episode_length = lambda: np.random.randint(min_episode_length, max_episode_length)
    episode_lengths = [new_episode_length() for _ in range(n_envs)]

    steps_left_in_episode = episode_lengths.copy()
    num_finished_episodes = np.zeros(n_envs)

    for step in tqdm.tqdm(range(n_updates), leave=False):
        # print(f"Step {step}")
        steps_since_last_update = np.zeros(n_envs)
        finished_an_episode_since_last_update = np.zeros(n_envs, dtype=bool)
        
        # Loop over all the envs, until all of them have produced a loss (reached
        # the end of an episode).
        while not all(finished_an_episode_since_last_update):
            # print(f"Episode lengths: {episode_lengths}")
            # print(f"Steps left: {steps_left_in_episode}")
            # print(f"Completed episodes: {num_finished_episodes}")
            # print(f"Used steps: {n_used_steps}")
            # print(f"Wasted steps: {n_wasted_steps}")

            for env in range(n_envs):
                
                if steps_left_in_episode[env] == 0:
                    # Perform the "backward()" for that env.
                    # This will use all steps since the last update (with grads).
                    n_used_steps += steps_since_last_update[env]
                    steps_since_last_update[env] = 0
                    
                    finished_an_episode_since_last_update[env] = True
                    num_finished_episodes[env] += 1
                    
                    # Sample the length of the next episode randomly.
                    length_of_next_episode = new_episode_length()
                    episode_lengths[env] = length_of_next_episode
                    steps_left_in_episode[env] = length_of_next_episode

                else:
                    steps_left_in_episode[env] -= 1
                    steps_since_last_update[env] += 1
            # breakpoint()
            
                    
        # Perform the "optimizer step" for the model.
        # This 'wastes' all the prediction tensors (actions) in unfinished episodes
        # because it would detach them.
        n_wasted_steps += int(steps_since_last_update.sum())
        # print(f"n episodes per env: {num_finished_episodes}")

    total_steps = n_used_steps + n_wasted_steps
    used_ratio = n_used_steps / total_steps
    wasted_ratio = n_wasted_steps / total_steps

    # print(f"Total steps: {total_steps}")
    # print(f"n_envs: {n_envs}")
    # print(f"n_updates: {n_updates}")
    # print(f"Used steps:   {n_used_steps} \t{used_ratio:.2%}")
    # print(f"Wasted steps: {n_wasted_steps} \t{used_ratio:.2%}")
    return n_used_steps, n_wasted_steps


if __name__ == "__main__":
        
    min_episode_length: int = 5
    max_episode_length: int = 100

    import matplotlib.pyplot as plt
    x = np.random.randint(1, 32, size=100)
    x.sort()
    used_ = []
    wasted_ = []

    for n_envs in tqdm.tqdm(x, desc="n_envs"):
        used, wasted = get_fraction_of_observations_with_grad(
            n_envs=n_envs,
            min_episode_length=min_episode_length,
            max_episode_length=max_episode_length,
            )
        used_.append(used)
        wasted_.append(wasted)

    y_used = np.array(used_)
    y_wasted = np.array(wasted)


    used_ratio = y_used / (y_used + y_wasted)
    fig: plt.Figure
    axes: List[plt.Axes]
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(f"Episode length ~ U[{min_episode_length},{max_episode_length}]")

    axes[0].set_title(f"Percentage of 'wasted' grads vs batch size")
    axes[0].scatter(x, used_ratio)
    axes[0].set_ylabel("% of 'wasted' gradients")
    axes[0].set_xlabel("batch size (number of environments)")

    used_per_env = y_used / n_envs

    axes[1].set_title(f"Data 'efficiency': total used steps per env")
    axes[1].scatter(x, used_per_env)
    axes[1].set_xlabel(f"# of environments")
    axes[1].set_ylabel(f"# of used steps per env")


    plt.show()
