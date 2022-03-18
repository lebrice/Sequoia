from typing import Callable, List

import numpy as np
import tqdm as tqdm


def get_fraction_of_observations_with_grad(
    n_envs: int,
    new_episode_length: Callable[[], int],
    n_updates: int = 10,
    min_episodes_before_update: int = 1,
):
    n_used_steps = 0
    n_wasted_steps = 0
    # min_episode_length = 0
    # max_episode_length = 10
    # n_envs = 10
    # new_episode_length = lambda: 10
    # The starting episode lengths for each env.
    # new_episode_length = lambda: 10
    # episode_lengths = [5, 10]
    # n_envs = 2
    episode_lengths = np.array([new_episode_length() for _ in range(n_envs)])
    steps_left_in_episode = episode_lengths.copy()
    num_finished_episodes = np.zeros(n_envs)

    for step in tqdm.tqdm(range(n_updates), leave=False):
        # print(f"Step {step}")
        steps_since_last_update = np.zeros(n_envs)
        finished_episodes_since_last_update = np.zeros(n_envs)

        # Loop over all the envs, until all of them have produced a loss (reached
        # the end of an episode).
        while not all(finished_episodes_since_last_update >= min_episodes_before_update):
            # print(f"Episode lengths: {episode_lengths}")
            # print(f"Steps left: {steps_left_in_episode}")
            # print(f"Completed episodes: {num_finished_episodes}")
            # print(f"Used steps: {n_used_steps}")
            # print(f"Wasted steps: {n_wasted_steps}")

            # print(steps_left_in_episode)
            for env in range(n_envs):
                if steps_left_in_episode[env] == 0:
                    # Perform the "backward()" for that env.
                    # This will use all steps since the last update (with grads).
                    used = steps_since_last_update[env]
                    n_used_steps += used
                    wasted = episode_lengths[env] - steps_since_last_update
                    # print(f"Step {step}, doing backward for env {env} using {used} steps.")
                    steps_since_last_update[env] = 0

                    finished_episodes_since_last_update[env] += 1
                    num_finished_episodes[env] += 1

                    # Sample the length of the next episode randomly.
                    length_of_next_episode = new_episode_length()
                    steps_left_in_episode[env] = length_of_next_episode
                else:
                    steps_left_in_episode[env] -= 1
                    steps_since_last_update[env] += 1

        # Perform the "optimizer step" for the model.
        # This 'wastes' all the prediction tensors (actions) in unfinished episodes
        # because it would detach them.
        wasted_per_env = steps_since_last_update
        n_wasted_steps += int(wasted_per_env.sum())
        # print(f"Updating model at step {step}, wasting {wasted_per_env} grads")
        # exit()
        # print(f"Ratio of used vs wasted so far: {n_used_steps}/{n_wasted_steps+n_used_steps}")
        # print(f"n episodes per env: {num_finished_episodes}")

    total_steps = n_used_steps + n_wasted_steps
    used_ratio = n_used_steps / total_steps
    wasted_ratio = n_wasted_steps / total_steps

    # print(f"Total steps: {total_steps}")
    # print(f"n_envs: {n_envs}")
    # print(f"n_updates: {n_updates}")
    # print(f"Used steps:   {n_used_steps} \t{used_ratio:.2%}")
    # print(f"Wasted steps: {n_wasted_steps} \t{wasted_ratio:.2%}")
    return n_used_steps, n_wasted_steps


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig: plt.Figure
    axes: List[plt.Axes]
    n_updates_per_run: int = 20
    fig, axes = plt.subplots(1, 2)
    import textwrap

    # x: np.ndarray = np.random.randint(1, 32, size=100)
    x: np.ndarray = np.arange(63, dtype=int) + 1

    min_episodes_before_update = 3
    # min_episodes_before_updates = [1, 3, 5]

    min_episode_length: int = 5
    max_episode_length: int = 100
    episode_len_dist = f"U[{min_episode_length},{max_episode_length}]"

    # Normally distributed episode lengths:
    # episode_length_mean = (max_episode_length + min_episode_length) / 2
    episode_length_mean = 50
    # episode_length_std = np.sqrt(max_episode_length - episode_length_mean)
    # episode_len_dist = f"N({episode_length_mean:.1f}, {episode_length_std:.1f})"
    episode_length_stds = [1.0, 3.0, 5.0, 10.0]
    episode_len_dist = f"N({episode_length_mean:.1f}, {episode_length_stds})"

    s = "s" if min_episodes_before_update > 1 else ""
    fig.suptitle(
        textwrap.dedent(
            f"""\
        Episode length ~ {episode_len_dist},
        Updating model when all envs have finished at least {min_episodes_before_update} episode{s},
        {n_updates_per_run} total updates per run.
        """
        )
    )

    # for min_episodes_before_update in min_episodes_before_updates:
    for episode_length_std in episode_length_stds:
        label = f"episode_length_std={episode_length_std:.1f}"
        # label = f"min_episodes_before_update={min_episodes_before_update}"

        # new_episode_length = lambda: np.random.randint(min_episode_length, max_episode_length)
        new_episode_length = lambda: int(np.random.normal(episode_length_mean, episode_length_std))

        # x.sort()
        used_ = []
        wasted_ = []

        for n_envs in tqdm.tqdm(x, desc="n_envs"):
            used, wasted = get_fraction_of_observations_with_grad(
                n_envs=n_envs,
                new_episode_length=new_episode_length,
                min_episodes_before_update=min_episodes_before_update,
                n_updates=n_updates_per_run,
            )
            used_.append(used)
            wasted_.append(wasted)

        y_used = np.array(used_)
        y_wasted = np.array(wasted_)

        used_ratio = y_used / (y_used + y_wasted)
        wasted_ratio = 1 - used_ratio

        axes[0].set_title(f"Percentage of used vs 'wasted' gradients w.r.t. batch size")
        axes[0].scatter(x, used_ratio, label=label)
        axes[0].set_ylim(0.0, 1.0)

        used_per_env = y_used / x / n_updates_per_run
        axes[1].scatter(x, used_per_env)

    fig.legend()
    # xs, ys = x, used_ratio
    # # zip joins x and y coordinates in pairs
    # for x_i, y_i in zip(xs, ys):
    #     label = f"({int(x_i)}, {y_i:.2f})"
    #     axes[0].annotate(label, # this is the text
    #                 (x_i, y_i), # this is the point to label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(0,10), # distance from text to points (x,y)
    #                 ha='center') # horizontal alignment can be left, right or center

    axes[0].set_ylabel("% of used gradients")
    axes[0].set_xlabel("batch size (number of environments)")

    axes[1].set_title(f"''Data efficiency'': Average number of used steps per update per env")

    axes[1].set_xlabel(f"# of environments")
    axes[1].set_ylabel(f"# of used steps per env")

    plt.show()
