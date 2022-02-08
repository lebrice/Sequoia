import time
from itertools import chain

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import wandb

from .config import Config, ExperimentType, HParams
from .data import get_loaders
from .evaluation import background_logistic, evaluate_features
from .losses import SimCLRLoss, class_sim_loss
from .models import Classifier, Encoder, Projector
from .utils import load_state, log, save_state


def train(
    encoder, projector, classifier, loader, optimizer, nt_xent_loss, epoch, hp: HParams, cfg: Config
) -> None:
    encoder.train()
    projector.train()
    classifier.train()

    losses, ce_losses, con_losses, class_losses = [], [], [], []

    for data, target in tqdm(loader, leave=False):
        if hp.experiment == ExperimentType.CLASSIFICATION:
            bs, c, h, w = data.size()
        else:
            bs, ncrops, c, h, w = data.size()
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        h = encoder(data.view((-1, c, h, w)))
        z = projector(h)

        if hp.experiment != ExperimentType.CLASSIFICATION:
            contrastive_loss = nt_xent_loss(z, hp.xent_temp)
            class_loss = class_sim_loss(z, target.view(-1), hp.xent_temp)
            con_losses.append(contrastive_loss.item())
            class_losses.append(class_loss.item())
        if hp.experiment not in {ExperimentType.CONTRASTIVE, ExperimentType.CLASS_CONTRASTIVE}:
            if hp.experiment == ExperimentType.SUCCESSIVE:
                logits = classifier(z)
            else:
                logits = classifier(h)
            classification_loss = F.cross_entropy(logits, target.view((-1)))
            ce_losses.append(classification_loss.item())

        if hp.experiment == ExperimentType.SUCCESSIVE or hp.experiment == ExperimentType.BRANCH:
            loss = classification_loss + contrastive_loss
        elif (
            hp.experiment == ExperimentType.CLASSIFICATION
            or hp.experiment == ExperimentType.AUGMENTED_CLASSIFICATION
        ):
            loss = classification_loss
        elif hp.experiment == ExperimentType.CONTRASTIVE:
            loss = contrastive_loss
        elif hp.experiment == ExperimentType.CLASS_CONTRASTIVE:
            loss = contrastive_loss + class_loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    log(
        {
            "train/avg_loss": np.mean(losses),
            "train/avg_con_loss": 0 if len(con_losses) == 0 else np.mean(con_losses),
            "train/avg_ce_loss": 0 if len(ce_losses) == 0 else np.mean(ce_losses),
            "train/avg_class_loss": 0 if len(class_losses) == 0 else np.mean(class_losses),
        },
        epoch,
        "train",
        cfg.log_wandb,
    )


def experiment(hp: HParams, cfg: Config):
    print("Experiment", hp.md5)
    print(torch.get_num_threads(), "cpu cores available")

    checkpoint = load_state(hp)
    if checkpoint is not None:
        print(f"Restoring training state from epoch {checkpoint['epoch']}")
        hp = checkpoint["hparams"]

    torch.manual_seed(1)

    # Dataset
    train_loader, train_eval_loader, test_loader = get_loaders(hp, cfg)

    # Models
    encoder = Encoder(hp)
    projector = Projector(hp)
    classifier = Classifier(hp)
    if checkpoint is not None:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        projector.load_state_dict(checkpoint["projector_state_dict"])
        classifier.load_state_dict(checkpoint["classifier_state_dict"])
    encoder = encoder.cuda()
    projector = projector.cuda()
    classifier = classifier.cuda()

    # Parametric Loss
    loss_fn = SimCLRLoss(hp.proj_dim, hp.use_bilinear_loss)
    if checkpoint is not None:
        loss_fn.load_state_dict(checkpoint["loss_state_dict"])
    loss_fn = loss_fn.cuda()

    # Optimizers and Schedulers
    init_lr = hp.max_lr / (hp.warmup_epochs + 1)
    optimizer = torch.optim.SGD(
        chain(
            encoder.parameters(),
            projector.parameters(),
            classifier.parameters(),
            loss_fn.parameters(),
        ),
        lr=init_lr,
        momentum=0.9,
        weight_decay=1e-6,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, hp.cooldown_epochs)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cosine_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Starting Epoch
    epoch = 1 if checkpoint is None else checkpoint["epoch"]

    # Background evaluation setup (has to be before wandb init in this process!)
    background_queue = None
    if cfg.evaluate_background:
        background_queue = mp.Queue()
        background_process = mp.Process(
            target=background_logistic, args=(hp, cfg, background_queue)
        )
        background_process.start()

    # Wandb
    if cfg.log_wandb:
        wandb.init(project="falr", config=hp.as_dict, group=hp.md5, job_type="main")

    for epoch in range(epoch, hp.warmup_epochs + hp.cooldown_epochs + 1):
        log({"train/lr": optimizer.param_groups[0]["lr"]}, epoch, "train", cfg.log_wandb)

        start = time.time()
        if hp.experiment == ExperimentType.CLASSIFICATION:
            train(
                encoder,
                projector,
                classifier,
                train_eval_loader,
                optimizer,
                loss_fn,
                epoch,
                hp,
                cfg,
            )
        else:
            train(encoder, projector, classifier, train_loader, optimizer, loss_fn, epoch, hp, cfg)
        log(
            {
                "train/runtime": time.time() - start,
            },
            epoch,
            "train",
            cfg.log_wandb,
        )

        if epoch <= hp.warmup_epochs:
            optimizer.param_groups[0]["lr"] = min(
                hp.max_lr, hp.max_lr * (epoch + 1) / (hp.warmup_epochs + 1)
            )  # Pytorch LambdaLR scheduler is buggy...
        elif hp.use_lr_decay:
            cosine_scheduler.step()

        if (
            (epoch == 1)
            or (epoch % cfg.evaluation_epoch_freq == 0)
            or (epoch == hp.warmup_epochs + hp.cooldown_epochs)
        ):
            evaluate_features(
                hp,
                encoder,
                projector,
                classifier,
                train_eval_loader,
                test_loader,
                epoch,
                cfg.log_wandb,
                background_queue,
            )
            if cfg.save_checkpoints:
                save_state(
                    hp,
                    cfg,
                    epoch + 1,
                    encoder,
                    projector,
                    classifier,
                    optimizer,
                    cosine_scheduler,
                    loss_fn,
                )

    if cfg.evaluate_background:
        background_queue.put(None)
        background_process.join()
