from pytorch_lightning import Trainer
from pl_bolts.models.rl import DQN

dqn = DQN("PongNoFrameskip-v4")
trainer = Trainer(gpus=1)
trainer.fit(dqn)
