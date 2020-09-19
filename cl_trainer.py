""" Variant of the Trainer class from pytorch-lightning more adapted for CL.
""" 
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from pytorch_lightning.loggers import LightningLoggerBase
from singledispatchmethod import singledispatchmethod
from torch.utils.data import DataLoader

from common.config import TrainerConfig
from common.loss import Loss
from settings import ClassIncrementalSetting, Results, Setting
from utils.logging_utils import get_logger


@dataclass
class CLTrainerOptions(TrainerConfig):
    
    def make_trainer(self,
                     loggers: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
                     callbacks: Optional[List[Callback]] = None) -> Trainer:
        """ Create a Trainer object from the command-line args.
        Adds the given loggers and callbacks as well.
        """
        return CLTrainer(
            logger=loggers,
            callbacks=callbacks,
            **self.to_dict()
        )


logger = get_logger(__file__)

class CLTrainer(Trainer):
    # TODO: customize the training procedure for CL, to make it easier on the
    # methods.
    # TODO: Also possible: create some 'required' Callbacks that notify the
    # model of a task switch, etc. 
    def fit(self,
            model: LightningModule,
            train_dataloader: Optional[DataLoader] = None,
            val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
            datamodule: Optional[Union[Setting, LightningDataModule]] = None):
        """ Changes the `fit` method slightly to allow training on a series of
        tasks.

        This is so we can learn one task at a time.
        """
        datamodule = (datamodule or
                      model.datamodule or
                      getattr(model, "setting", None))
        if isinstance(datamodule, Setting):
            setting: Setting = datamodule
            logger.debug(f"Training the model of type {type(model)} on a setting of type {type(datamodule)}!")
            self.fit_setting(setting, model)
        else:
            logger.info(
                f"datamodule {datamodule} isn't an instance of `Setting`, "
                f"defaulting back to using `super().fit`.")
            return super().fit(
                model,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
            )

    @singledispatchmethod
    def fit_setting(self, setting: Setting, model: LightningModule):
        logger.info(f"No known custom training procedure for setting of type {type(setting)}.")
        logger.info(f"Defaulting back to super().fit(model, datamodule=setting)")
        super().fit(model=model, datamodule=setting)

    def test(self,
             model: Optional[LightningModule] = None,
             test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
             ckpt_path: Optional[str] = "best",
             verbose: bool = True,
             datamodule: Optional[LightningDataModule] = None):
        """ BUG: Writing this because Trainer.test doesn't return anything, for
        some reason!

        TODO: Not sure we really need this. This sort-of blurs the line between
        what is the Trainer vs the Setting's responsability w.r.t. the
        evaluation loop.
        """
        datamodule = (datamodule or
                      self.get_model().datamodule or
                      getattr(self.get_model(), "setting", None))

        if isinstance(datamodule, Setting):
            setting: Setting = datamodule
            # NOTE: This is subtle, but self.model is often a
            # LightningDataParallel object, while self.get_model() returns the
            # wrapped LightningModule. If we want to perform parallel inference,
            # then we wouldn't we need to inform all the models in the
            # LightningDataParallel of a task switch, when applicable?
            logger.debug(
                f"Testing the model on a setting of type {type(setting)}!"
            )
            return self.test_setting(
                setting,
                model=model,
                ckpt_path=ckpt_path,
                verbose=verbose,
            )
        logger.info(
            f"datamodule {datamodule} isn't an instance of `Setting`, "
            f"defaulting back to using `super().test`."
        )
        return super().test(
            model=model,
            test_dataloaders=test_dataloaders,
            ckpt_path=ckpt_path,
            verbose=verbose,
            datamodule=datamodule,
        )

    @singledispatchmethod
    def test_setting(self,
                     setting: Setting,
                     model: LightningModule,
                     ckpt_path: Optional[str] = 'best',
                     verbose: bool = True,) -> Results:
        """Tests the method and returns the Results.

        Overwrite this or register your own testing method to customize testing
        for your experimental setting.

        Returns:
            Results: A Results object for this particular setting.
        """
        logger.info(f"No registered custom testing procedure for setting of type {type(setting)}.")
        logger.info(f"Defaulting back to super().test(model, datamodule=setting)")
        assert self.datamodule is setting
        super().test(
            datamodule=setting,
            ckpt_path=ckpt_path,
            verbose=verbose
        )

    @fit_setting.register
    def fit_class_incremental(self, setting: ClassIncrementalSetting, model: LightningModule):
        n_tasks = setting.nb_tasks
        logger.info(f"Number of tasks: {n_tasks}")
        logger.info(f"Number of classes in task: {setting.num_classes}")

        for i in range(n_tasks):
            logger.info(f"Starting training on task #{i}")
            setting.current_task_id = i

            if setting.task_labels_at_train_time:
                # This is always true in the ClassIncremental & TaskIncremental
                # settings for now.
                # TODO: @lebrice What should we call 'on_task_switch' on? the Method? the Model?
                if hasattr(model, "on_task_switch") and callable(model.on_task_switch):
                    model.on_task_switch(i)
            super().fit(
                model,
                datamodule=setting,
            )

    @test_setting.register
    def test_class_incremental(self,
                            setting: ClassIncrementalSetting,
                            model: LightningModule,
                            ckpt_path: Optional[str] = 'best',
                            verbose: bool = True,):
        """Tests the method and returns the Results.

        Overwrite this to customize testing for your experimental setting.

        Returns:
            Results: A Results object for this particular setting.
        """

        if ckpt_path == "best":
            logger.warning(UserWarning(
                "When evaluating on a Continual Learning Setting, it might not be "
                "a good idea to use a value of 'best' for the 'ckpt_path' "
                "argument, because if the model had its best validation "
                "accuracy during the first task (as is usually the case), then "
                "before testing, its weights will be reloaded from the "
                "corresponding checkpoint, and the model will suck at all "
                "later tasks! "
            ))
            logger.warning(f"Setting ckpt_path to None for now.")
            ckpt_path = None
        model = model or self.get_model()
        assert self.datamodule is setting


        results: List[Dict] = []
        for task_id in range(setting.nb_tasks):
            logger.info(f"Starting evaluation on task {task_id}.")
            setting.current_task_id = task_id
            if setting.task_labels_at_test_time:
                model.on_task_switch(task_id)
            task_results = super().test(
                # model=model,
                # datamodule=setting,
                # verbose=False,
            )
            assert False, task_results
            assert task_results is not None, "Trainer.test() returned None?!!"
            results.append(task_results)
        
        return results


        test_dataloaders = setting.test_dataloaders()
        model = model or self.get_model()
        # TODO: Here we are 'manually' evaluating on one test dataset at a time.
        # However, in pytorch_lightning, if a LightningModule's
        # `test_dataloader` method returns more than a single dataloader, then
        # the Trainer takes care of evaluating on each dataset in sequence.
        # Here we basically do this manually, so that the trainer doesn't give
        # the `dataloader_idx` keyword argument to the eval_step() method on the
        # LightningModule.
        
        # Create a list of the total test Loss for each task.
        task_losses: List[Loss] = []
        # TODO: Fix this evaluation loop.

        for i, task_loader in enumerate(test_dataloaders):
            logger.info(f"Starting evaluation on task {i}.")
            setting.current_task_id = i

            if setting.task_labels_at_test_time:
                model.on_task_switch(i)

            # BUG: super().test never returns anything, so Idk why, but this
            # here seems to work as a replacement atm.
            results = super().test(
                # model=model,
                test_dataloaders=task_loader,
            )
            # # TODO: Remove this. Maybe even move this into the module of the
            # # corresponding setting.
            assert False, f"Results: {results}"
            eval_loop_results, eval_results = self.run_evaluation(test_mode=True)
            assert isinstance(eval_results, list)
            assert len(eval_results) == 1
            result = eval_results[0]
            assert "loss_object" in result
            task_loss = result["loss_object"]
            task_losses.append(task_loss)

        return task_losses
