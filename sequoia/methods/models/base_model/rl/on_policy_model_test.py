from .on_policy_model import OnPolicyModel
import gym
import torch
from pytorch_lightning.trainer import Trainer
import itertools
import pytest


def test_cartpole_manual():
    # env = gym.vector.make("CartPole-v0", num_envs=2, asynchronous=False)
    # val_env = gym.vector.make("CartPole-v0", num_envs=2, asynchronous=False)
    env = gym.make("CartPole-v0")
    val_env = gym.make("CartPole-v0")
    train_seed = 123
    val_seed = 456
    from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors

    # seed everything.
    from pytorch_lightning.utilities.seed import seed_everything

    seed_everything(123)

    env.seed(train_seed)
    env.action_space.seed(train_seed)
    env.observation_space.seed(train_seed)

    val_env.seed(val_seed)
    val_env.action_space.seed(val_seed)
    val_env.observation_space.seed(val_seed)

    # Note: Have to wrap the env so it works for the model.
    from sequoia.settings.rl.wrappers import TypedObjectsWrapper
    from .on_policy_model import (
        OnPolicyModel,
        Observation,
        DiscreteActionBatch,
        DiscreteAction,
        Rewards,
    )

    # train_env = TypedObjectsWrapper(env=env, observations_type=Observation, actions_type=DiscreteAction, rewards_type=Rewards)
    # train_env = ConvertToFromTensors(train_env, device="cpu")

    model = OnPolicyModel(train_env=env, val_env=val_env)
    optimizer = model.configure_optimizers()
    train_dl = model.train_dataloader()

    episodes_per_update = 3
    max_episodes = 10

    # TODO: The `with_is_last` thingy over the top of the DataLoader is still causing issues!
    # There's a delay of one step between the dataloader and the model. This is annoying.
    # Could probably create an 'adapter' of some sort that recomputes the forward pass of the actions
    # for just that single misaligned step?
    model.n_policy_updates = 0
    for i, episode in enumerate(itertools.islice(train_dl, max_episodes)):
        step_output = model.training_step(episode, batch_idx=i)
        loss = step_output["loss"]

        is_update_step = episodes_per_update == 1 or (i > 0 and i % episodes_per_update == 0)
        
        loss.backward(retain_graph=not is_update_step)

        print(step_output)

        assert set(episode.model_versions) == {model.n_policy_updates}

        if is_update_step:
            print(f"Update #{model.n_policy_updates} at step {i}")

            # model.optimizer_step()
            optimizer.step()

            optimizer.zero_grad()
            # Udpate the 'deployed' policy.
            train_dl.send(model)
            model.n_policy_updates += 1

        assert i < max_episodes


# import brax
# from brax.envs import _envs
# from gym.envs.registration import register
# import brax
# from brax.envs import _envs
# from brax.envs import create_gym_env
# from functools import partial
# @pytest.fixture(scope="session")
# def register_brax_envs():
#     for k, v in _envs.items():
#         register(f"brax.{k}-v0", entry_point=partial(create_gym_env, k))


# def test_brax_pendulum(register_brax_envs):
#     env = create_gym_env("inverted_pendulum", batch_size=10, seed=123)
#     # env = gym.make("brax.inverted_pendulum-v0", batch_size=10, seed=123)
#     print(env.reset())

#     assert False


@pytest.mark.timeout(5)
@pytest.mark.parametrize("train_seed", [123, 222])
@pytest.mark.parametrize("recompute_forward_passes", [True, False])
def test_cartpole_pl(train_seed: int, recompute_forward_passes: bool):
    env = gym.make("CartPole-v0")
    val_seed = 456
    # seed everything.
    env.seed(train_seed)
    env.action_space.seed(train_seed)
    env.observation_space.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    val_env = gym.make("CartPole-v0")
    val_env.seed(val_seed)
    val_env.action_space.seed(val_seed)
    val_env.observation_space.seed(val_seed)

    episodes_per_update = 3
    episodes_per_epoch = 100
    episodes_per_val_epoch = 10

    model = OnPolicyModel(
        train_env=env,
        val_env=val_env,
        episodes_per_train_epoch=episodes_per_epoch,
        episodes_per_val_epoch=episodes_per_val_epoch,
        recompute_forward_passes=recompute_forward_passes,
    )

    trainer = Trainer(
        max_epochs=1,
        accumulate_grad_batches=episodes_per_update,
        checkpoint_callback=False,
        logger=False,
    )
    trainer.fit(model)
    n_updates = model.global_step

    assert model.n_training_steps == episodes_per_epoch
    if recompute_forward_passes:
        # We are recomputing the first episode after each update.
        assert model.recomputed_forward_passes == n_updates
        assert model.wasted_forward_passes == 0
    else:
        # We are 'wasting'' the first episode after each model update.
        assert model.recomputed_forward_passes == 0
        assert model.wasted_forward_passes == n_updates

    assert model.n_validation_steps == episodes_per_val_epoch
    assert model.global_step == episodes_per_epoch // episodes_per_update
    assert model.n_policy_updates == n_updates

    # NOTE: Now need to add metrics into the log dict.
    # assert False, dict(
    #     n_training_steps=model.n_training_steps,
    #     wasted_forward_passes=model.wasted_forward_passes,
    #     recomputed_forward_passes=model.recomputed_forward_passes,
    #     n_forward_passes=model.n_forward_passes,
    #     global_step=model.global_step,
    #     n_policy_updates=model.n_policy_updates,
    # )
