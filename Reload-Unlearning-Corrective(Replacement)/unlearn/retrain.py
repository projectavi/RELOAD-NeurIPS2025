import os

import torch
import wandb
from trainer import *

import utils

from .impl import iterative_unlearn


def retrain(data_loaders, model, criterion, optimizer, epoch, args, mask, wandb_run):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]

    args.unlearn_lr = args.lr

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed
    for epoch in range(0, args.unlearn_epochs):
        print("Starting epoch: ", epoch)
        evals = train(retain_loader, model, criterion, optimizer, epoch, args, forget_loader)

        r_loss.update(evals["train_loss"])
        r_accuracy.update(evals["train_accuracy"])
        f_loss.update(evals["forget_loss"])
        f_accuracy.update(evals["forget_accuracy"])

        wandb_run.log({
            "Retrain In Process Forget Loss": evals["forget_loss"],
            "Retrain In Process Forget Accuracy": evals["forget_accuracy"],
            "Retrain In Process Retain Loss": evals["train_loss"],
            "Retrain In Process Retain Accuracy": evals["train_accuracy"],
        })

        scheduler.step()

    evals = {
        "train": {"loss": r_loss, "accuracy": r_accuracy},
        "forget": {"loss": f_loss, "accuracy": f_accuracy},
        "retain": {"loss": r_loss, "accuracy": r_accuracy},
    }

    return model, evals
