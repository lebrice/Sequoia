# TODO: Just testing out the 'REINFORCE' implementation from the reinforce_head.py
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

from methods.models.output_heads.rl.reinforce_head import (
    ClassificationOutput, ReinforceHead)


def main():
    env = gym.make('CartPole-v0')
    policy_net = ReinforceHead(
        input_size=4,
        action_space=env.action_space,
        reward_space=spaces.Box(0, 1, shape=())
    ).cuda()
    max_episode_num = 5000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []


        optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)

        for steps in range(max_steps):
            env.render()

            observations = None
            representations = torch.as_tensor(state).cuda()

            actions: ClassificationOutput = policy_net(observations, representations)
            # NOTE: Index of 0 since there is no batching.
            action, log_prob = actions.actions_np[0], actions.y_pred_log_prob[0]
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                optimizer.zero_grad()
                policy_gradient = policy_net.get_policy_gradient_for_episode(rewards, log_probs)
                policy_gradient.backward()
                optimizer.step()

                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    print(
                        f"episode: {episode}, "
                        f"total reward: {np.sum(rewards):.3f}, "
                        f"average_reward: {np.mean(all_rewards[-10:]):.3f}, "
                        f"length: {steps}"
                    )
                break

            state = new_state
        
    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()

if __name__ == "__main__":
    main()
