import gym
import gym.vector
import torch
from pytorch_lightning import Trainer
from sequoia.methods.models.base_model.rl.on_policy_model import (
    OnPolicyModel,
    WhatToDoWithOffPolicyData,
)
import logging

logging.getLogger().setLevel(logging.DEBUG)


def seed_env(env: gym.Env, seed: int) -> None:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def main():
    num_envs = 10
    train_seed = 123
    max_epochs = 10
    episodes_per_update = 1
    episodes_per_epoch = 100
    episodes_per_val_epoch = 10

    env = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=False)
    val_env = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=False)
    test_env = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=False)

    val_seed = train_seed * 2
    test_seed = train_seed * 3

    # seed everything.
    from pytorch_lightning.utilities.seed import seed_everything

    seed_everything(train_seed)

    seed_env(env, train_seed)
    seed_env(val_env, val_seed)
    seed_env(test_env, test_seed)

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
        gpus=torch.cuda.device_count(),
        accumulate_grad_batches=1,
    )
    trainer.fit(model)
    test_results = trainer.test(model)
    print(test_results)


if __name__ == "__main__":
    main()
