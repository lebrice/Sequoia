from .base_model_rl import BaseRLModel
import gym
import torch
from pytorch_lightning.trainer import Trainer
import itertools


def test_cartpole_manual():
    env = gym.make("CartPole-v0")
    
    # seed everything.
    env.seed(123)
    env.action_space.seed(123)
    env.observation_space.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    model = BaseRLModel(env)
    optimizer = model.configure_optimizers()
    # act = model(env.reset(), action_space=env.action_space)
    # assert False, act
    train_dl = model.train_dataloader()
    episodes_per_update = 3
    max_episodes = 10

    # TODO: The `with_is_last` thingy over the top of the DataLoader is still causing issues!
    # There's a delay of one step between the dataloader and the model. This is annoying.
    # Could probably create an 'adapter' of some sort that recomputes the forward pass of the actions
    # for just that single misaligned step?
    model.n_updates = 0
    for i, episode in enumerate(itertools.islice(train_dl, max_episodes)):
        loss = model.training_step(episode, batch_idx=i)

        is_update_step = episodes_per_update == 1 or (i > 0 and i % episodes_per_update == 0)
        loss.backward(retain_graph=not is_update_step)

        print(loss)

        assert set(episode.model_versions) == {model.n_updates}

        if is_update_step:
            print(f"Update #{model.n_updates} at step {i}")

            # model.optimizer_step()
            optimizer.step()

            optimizer.zero_grad()
            # Udpate the 'deployed' policy.
            train_dl.send(model)
            model.n_updates += 1

        assert i < max_episodes

import pytest
@pytest.mark.timeout(5)
def test_cartpole_pl():
    env = gym.make("CartPole-v0")
    
    # seed everything.
    env.seed(123)
    env.action_space.seed(123)
    env.observation_space.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    episodes_per_update = 3
    max_episodes = 10
    model = BaseRLModel(env)
    trainer = Trainer(
        max_steps=max_episodes,
        min_steps = 1,
        accumulate_grad_batches=episodes_per_update,
        # max_time: ,
    )
    trainer.fit(model)

    assert False, (model.recomputed_forward_passes, model.n_forward_passes, model.global_step, model.n_updates)
