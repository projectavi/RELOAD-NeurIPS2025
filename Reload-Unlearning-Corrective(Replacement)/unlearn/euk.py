import sys
import time

import torch
import utils
import wandb
from trainer import *

from .impl import iterative_unlearn

sys.path.append(".")
import copy

import torch.nn as nn

def getRetrainLayers(m, name, ret):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        ret.append((m, name))
        #print(name)
    for child_name, child in m.named_children():
        getRetrainLayers(child, f'{name}.{child_name}', ret)
    return ret

def _reinit(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def resetFinalResnet(model, num_retrain, modname, logger, reinit=True, pretrain_path=None):
    for param in model.parameters():
        param.requires_grad = False

    done = 0
    ret = getRetrainLayers(model, 'M', [])
    ret.reverse()
    for idx in range(len(ret)):
        if reinit:
            if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
                _reinit(ret[idx][0])
        if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
            done += 1
        for param in ret[idx][0].parameters():
            param.requires_grad = True
        if done >= num_retrain:
            break

    return model

MAX_LOSS = 1E8

def EU_k(data_loaders, model, criterion, optimizer, epoch, args, mask=None, wandb_run=None):

    forget_loader = data_loaders["full"] if args.task == "replace" else data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    valid_loader_full = data_loaders["test"]

    K = 3 # Default in repo is lower (1) and upper (3), originally 2 was selected as midpoint
    args.unlearn_epochs = 62 # Number of epochs used for initial training

    model = resetFinalResnet(model, K, args.arch, None, reinit=True)  # Turns parameters to freeze off, reinitializes rest
    model = model.to(args.device)

    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    t_loss = utils.AverageMeter()
    t_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed

    for epoch in range(1, args.unlearn_epochs + 1):

        print("==> EU-K unlearning ...")

        evals = train(retain_loader, model, criterion, optimizer, epoch, args, forget_loader)

        r_loss.update(evals["train_loss"])
        r_accuracy.update(evals["train_accuracy"])
        f_loss.update(evals["forget_loss"])
        f_accuracy.update(evals["forget_accuracy"])


    evaluations = {
        "train": {"loss": t_loss, "accuracy": t_accuracy},
        "forget": {"loss": f_loss, "accuracy": f_accuracy},
        "retain": {"loss": r_loss, "accuracy": r_accuracy},
        "epochs": args.unlearn_epochs,
    }

    return model, evaluations