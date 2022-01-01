import logging

from pytorch_lightning.utilities import seed
from .on_policy_model import OnPolicyModel, WhatToDoWithOffPolicyData
import gym
import torch
from pytorch_lightning.trainer import Trainer
import itertools
import pytest
from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors
from pytorch_lightning.utilities.seed import seed_everything
from sequoia.conftest import param_requires_cuda


def seed_env(env: gym.Env, seed: int) -> None:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def test_cartpole_manual(monkeypatch):
    # env = gym.vector.make("CartPole-v0", num_envs=2, asynchronous=False)
    # val_env = gym.vector.make("CartPole-v0", num_envs=2, asynchronous=False)
    env = gym.make("CartPole-v0")
    val_env = gym.make("CartPole-v0")
    train_seed = 123
    val_seed = 456

    # seed everything.

    seed_everything(123)
    seed_env(env, train_seed)
    seed_env(val_env, val_seed)

    # Note: Have to wrap the env so it works for the model.
    from sequoia.settings.rl.wrappers import TypedObjectsWrapper
    from .on_policy_model import (
        OnPolicyModel,
        Observation,
        DiscreteAction,
        Rewards,
    )

    episodes_per_update = 3
    max_updates = 10

    model = OnPolicyModel(
        train_env=env,
        val_env=val_env,
        hparams=OnPolicyModel.HParams(
            episodes_per_update=episodes_per_update,
            what_to_do_with_off_policy_data=WhatToDoWithOffPolicyData.redo_forward_pass,
        ),
    )
    optimizer = model.configure_optimizers()
    train_dl = model.train_dataloader()

    monkeypatch.setattr(
        OnPolicyModel,
        "global_step",
        property(
            fget=lambda self: self._global_step,
            fset=lambda self, val: setattr(self, "_global_step", val),
        ),
    )
    for i, episodes in enumerate(itertools.islice(train_dl, max_updates)):
        model._global_step = i
        step_output = model.training_step(episodes, batch_idx=i)
        assert step_output is not None
        loss = step_output["loss"]
        loss.backward()

        for i, episode in enumerate(episodes):
            # NOTE: Not quite true, actually. Have to think about this again.
            assert set(episode.model_versions) == {model.global_step}, i

        optimizer.step()
        optimizer.zero_grad()
        # Update the 'deployed' policy.
        train_dl.send(model)
        model.n_policy_updates += 1


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


@pytest.mark.timeout(20)
@pytest.mark.parametrize("train_seed", [123, 222])
@pytest.mark.parametrize("recompute_forward_passes", [True, False])
@pytest.mark.parametrize("use_gpus", [False, param_requires_cuda(True)])
def test_cartpole_pl(train_seed: int, recompute_forward_passes: bool, use_gpus: bool):
    env = gym.make("CartPole-v0")
    val_env = gym.make("CartPole-v0")
    test_env = gym.make("CartPole-v0")

    val_seed = train_seed * 2
    test_seed = train_seed * 3

    # seed everything.
    from pytorch_lightning.utilities.seed import seed_everything

    seed_everything(train_seed)
    seed_env(env, train_seed)
    seed_env(val_env, val_seed)
    seed_env(test_env, test_seed)

    max_epochs = 1
    episodes_per_update = 10
    episodes_per_epoch = 100
    episodes_per_val_epoch = 100
    # from .on_policy_model import logger
    # logger.setLevel(logging.DEBUG)
    model = OnPolicyModel(
        train_env=env,
        val_env=val_env,
        test_env=test_env,
        episodes_per_train_epoch=episodes_per_epoch,
        episodes_per_val_epoch=episodes_per_val_epoch,
        hparams=OnPolicyModel.HParams(
            episodes_per_update=episodes_per_update,
            what_to_do_with_off_policy_data=WhatToDoWithOffPolicyData.redo_forward_pass,
        ),
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        checkpoint_callback=False,
        logger=False,
        gpus=torch.cuda.device_count() if use_gpus else None,
    )
    trainer.fit(model)

    n_updates = model.global_step
    if recompute_forward_passes:
        # We are recomputing the first episode after each update.
        assert model.n_recomputed_forward_passes == n_updates, (
            model.n_on_policy_episodes,
            model.n_recomputed_forward_passes,
        )
        assert model.n_wasted_forward_passes == 0
    else:
        # We are 'wasting'' the first episode after each model update.
        assert model.n_recomputed_forward_passes == 0
        assert model.n_wasted_forward_passes == n_updates

    # NOTE: Why 2 in sanity check?
    from pytorch_lightning.trainer.states import RunningStage

    assert model.steps_per_trainer_stage == {
        RunningStage.SANITY_CHECKING: 2,
        RunningStage.TRAINING: model.episodes_per_train_epoch * max_epochs
        - model.n_wasted_forward_passes,
        RunningStage.VALIDATING: model.episodes_per_val_epoch * max_epochs,
    }
    assert model.global_step == episodes_per_epoch // episodes_per_update
    assert model.n_policy_updates == n_updates

    test_results = trainer.test(model)
    print(test_results)

    # NOTE: The number of test steps == number of val steps per epoch atm.
    episodes_per_test_epoch = model.episodes_per_val_epoch
    assert model.steps_per_trainer_stage[RunningStage.TESTING] == episodes_per_test_epoch
    # NOTE: Now need to add metrics into the log dict.
    # assert False, dict(
    #     n_training_steps=model.n_training_steps,
    #     wasted_forward_passes=model.wasted_forward_passes,
    #     recomputed_forward_passes=model.recomputed_forward_passes,
    #     n_forward_passes=model.n_forward_passes,
    #     global_step=model.global_step,
    #     n_policy_updates=model.n_policy_updates,
    # )


@pytest.mark.parametrize("num_envs", [1, 2, 5])
def test_cartpole_vecenv_manual(num_envs: int):
    env = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=False)
    val_env = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=False)

    train_seed = 123
    val_seed = 456

    # seed everything.

    seed_everything(123)
    seed_env(env, train_seed)
    seed_env(val_env, val_seed)

    # Note: Have to wrap the env so it works for the model.
    from .on_policy_model import (
        OnPolicyModel,
        Observation,
        DiscreteAction,
        DiscreteAction,
        Rewards,
    )

    model = OnPolicyModel(train_env=env, val_env=val_env)
    optimizer = model.configure_optimizers()
    train_dl = model.train_dataloader()
    episodes_per_update = 3
    max_updates = 10

    assert train_dl.batch_size == episodes_per_update

    # TODO: The `with_is_last` thingy over the top of the DataLoader is still causing issues!
    # There's a delay of one step between the dataloader and the model. This is annoying.
    # Could probably create an 'adapter' of some sort that recomputes the forward pass of the actions
    # for just that single misaligned step?
    assert model.n_policy_updates == 0

    for i, episodes in enumerate(itertools.islice(train_dl, max_updates)):
        step_output = model.training_step(episodes, batch_idx=i)
        assert step_output is not None
        loss = step_output["loss"]

        print(i, [episode.model_versions for episode in episodes])
        loss.backward()

        print(step_output)
        for episode in episodes:
            assert set(episode.model_versions) == {model.n_policy_updates}

        print(f"Update #{model.n_policy_updates} at step {i}")

        # model.optimizer_step()
        optimizer.step()

        optimizer.zero_grad()
        # Udpate the 'deployed' policy.
        train_dl.send(model)

        model.n_policy_updates += 1

        assert i < max_updates


@pytest.mark.timeout(5)
@pytest.mark.parametrize("train_seed", [123, 222])
@pytest.mark.parametrize("recompute_forward_passes", [True, False])
@pytest.mark.parametrize("num_envs", [1, 2, 3])
@pytest.mark.parametrize("use_gpus", [False, param_requires_cuda(True)])
def test_vecenv_cartpole_pl(
    train_seed: int, recompute_forward_passes: bool, num_envs: int, use_gpus: bool
):
    """ TODO: There is still a bug with PL about tryign to call backward twice, so we need to make a more fine-grained tests as in above perhaps. """
    env = gym.vector.make("CartPole-v0", num_envs=num_envs)
    val_env = gym.vector.make("CartPole-v0", num_envs=num_envs)
    test_env = gym.vector.make("CartPole-v0", num_envs=num_envs)

    val_seed = train_seed * 2
    test_seed = train_seed * 3

    # seed everything.
    from pytorch_lightning.utilities.seed import seed_everything

    seed_everything(train_seed)

    seed_env(env, train_seed)
    seed_env(val_env, val_seed)
    seed_env(test_env, test_seed)

    max_epochs = 1
    episodes_per_update = 3
    episodes_per_epoch = 10
    episodes_per_val_epoch = 10
    # from .on_policy_model import logger
    # logger.setLevel(logging.DEBUG)
    model = OnPolicyModel(
        train_env=env,
        val_env=val_env,
        test_env=test_env,
        episodes_per_train_epoch=episodes_per_epoch,
        episodes_per_val_epoch=episodes_per_val_epoch,
        recompute_forward_passes=recompute_forward_passes,
        hparams=OnPolicyModel.HParams(episodes_per_update=episodes_per_update),
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        checkpoint_callback=False,
        logger=False,
        gpus=torch.cuda.device_count() if use_gpus else None,
    )
    trainer.fit(model)

    n_updates = model.global_step
    if recompute_forward_passes:
        # We are recomputing the first episode after each update.
        assert model.n_recomputed_forward_passes == n_updates
        assert model.n_wasted_forward_passes == 0
    else:
        # We are 'wasting'' the first episode after each model update.
        assert model.n_recomputed_forward_passes == 0
        assert model.n_wasted_forward_passes == n_updates

    # NOTE: Why 2 in sanity check?
    from pytorch_lightning.trainer.states import RunningStage

    assert model.steps_per_trainer_stage == {
        RunningStage.SANITY_CHECKING: 2,
        RunningStage.TRAINING: model.episodes_per_train_epoch * max_epochs
        - model.n_wasted_forward_passes,
        RunningStage.VALIDATING: model.episodes_per_val_epoch * max_epochs,
    }
    assert model.global_step == episodes_per_epoch // episodes_per_update
    assert model.n_policy_updates == n_updates

    test_results = trainer.test(model)
    print(test_results)

    # NOTE: The number of test steps == number of val steps per epoch atm.
    episodes_per_test_epoch = model.episodes_per_val_epoch
    assert model.steps_per_trainer_stage[RunningStage.TESTING] == episodes_per_test_epoch

