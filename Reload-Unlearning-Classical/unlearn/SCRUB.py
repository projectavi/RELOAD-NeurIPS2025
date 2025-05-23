import sys
import time

import torch
import utils
import wandb

from .impl import iterative_unlearn

sys.path.append(".")
import copy

import torch.nn as nn
from .thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from .thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss

from .thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate

def SCRUB(data_loaders, model, criterion, optimizer, epoch, args, mask, wandb_run):

    forget_loader = data_loaders["full"] if args.task == "replace" else data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    valid_loader_full = data_loaders["test"]

    # FROM REFERENCE IMPLEMENTATION
    optim = 'adam'
    gamma = 1
    alpha = 0.5
    beta = 0
    smoothing = 0.5
    msteps = 3
    clip = 0.2
    sstart = 10
    kd_T = 2
    distill = 'kd'

    sgda_epochs = 10
    sgda_learning_rate = 0.0005
    lr_decay_epochs = [5,8,9]
    lr_decay_rate = 0.1
    sgda_weight_decay = 0.1#5e-4
    sgda_momentum = 0.9

    args.optim = optim
    args.gamma = gamma
    args.alpha = alpha
    args.beta = beta
    args.smoothing = smoothing
    args.msteps = msteps
    args.clip = clip
    args.sstart = sstart
    args.kd_T = kd_T
    args.distill = distill

    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(trainable_list.parameters(),
                            lr=sgda_learning_rate,
                            momentum=sgda_momentum,
                            weight_decay=sgda_weight_decay)
    elif args.optim == "adam": 
        optimizer = torch.optim.Adam(trainable_list.parameters(),
                            lr=sgda_learning_rate,
                            weight_decay=sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = torch.optim.RMSprop(trainable_list.parameters(),
                            lr=sgda_learning_rate,
                            momentum=sgda_momentum,
                            weight_decay=sgda_weight_decay)

    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    t_loss = utils.AverageMeter()
    t_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    acc_rs = []
    acc_fs = []
    acc_ts = []
    acc_vs = []
    for epoch in range(1, sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, args, lr_decay_epochs, lr_decay_rate, sgda_learning_rate, optimizer)

        print("==> SCRUB unlearning ...")

        acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
        acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
        acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())
        acc_vs.append(100 - acc_v.item())

        r_loss.update(loss_r, 32)
        f_loss.update(loss_f, 32)
        r_accuracy.update(100 - acc_r.item(), 32)
        f_accuracy.update(100 - acc_f.item())

        maximize_loss = 0
        if epoch <= args.msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer, args,
                                          "maximize")
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, criterion_list, optimizer, args,
                                              "minimize")

        t_loss.update(train_loss, 32)
        t_accuracy.update(train_acc, 32)

        print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss,
                                                                                     train_acc))
    acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
    acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
    acc_rs.append(100 - acc_r.item())
    acc_fs.append(100 - acc_f.item())
    acc_vs.append(100 - acc_v.item())

    evaluations = {
        "train": {"loss": t_loss, "accuracy": t_accuracy},
        "forget": {"loss": f_loss, "accuracy": f_accuracy},
        "retain": {"loss": r_loss, "accuracy": r_accuracy},
        "epochs": sgda_epochs,
    }

    return model_s, evaluations