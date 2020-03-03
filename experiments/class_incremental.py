from dataclasses import dataclass

from simple_parsing import field, choice

from experiments.self_supervised import SelfSupervised
from config import Config

@dataclass
class ClassIncremental(SelfSupervised):
    config: Config = Config(class_incremental=True)

    def run(self):
        # TODO: measure the loss and accuracy on a per-class basis?
        train_epoch_loss: List[LossInfo] = []
        valid_epoch_loss: List[LossInfo] = []
        for epoch in range(self.hparams.epochs):
            for train_loss in self.train_iter(epoch, self.train_loader):
                print(train_loss.tensors.keys())
                exit()

            train_epoch_loss.append(train_loss)
            
            for valid_loss in self.test_iter(epoch, self.valid_loader):
                pass
            valid_epoch_loss.append(valid_loss)

            if self.reconstruction_task:
                with torch.no_grad():
                    sample = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
                    sample = sample.cpu().view(64, 1, 28, 28)
                    save_image(sample, os.path.join(self.config.log_dir, f"sample_{epoch}.png"))

            if self.config.wandb:
                # TODO: do some nice logging to wandb?:
                wandb.log(TODO)
        
        import matplotlib.pyplot as plt
        fig: plt.Figure = plt.figure()
        plt.plot([loss.total_loss for loss in train_epoch_loss], label="train_loss")
        plt.plot([loss.total_loss for loss in valid_epoch_loss], label="valid_loss")
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_loss.jpg"))


        fig: plt.Figure = plt.figure()
        plt.plot([loss.metrics.accuracy for loss in train_epoch_loss], label="train_accuracy")
        plt.plot([loss.metrics.accuracy for loss in valid_epoch_loss], label="valid_accuracy")
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_accuracy.jpg"))

    
