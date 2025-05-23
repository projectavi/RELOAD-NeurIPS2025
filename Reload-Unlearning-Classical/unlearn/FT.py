import sys
import time

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")

from trainer import validate


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def FT_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, with_l1=False
):
    train_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]

    train_loss = utils.AverageMeter()
    train_accuracy = utils.AverageMeter()

    forget_losses = utils.AverageMeter()
    forget_accuracies = utils.AverageMeter()

    # switch to train mode
    model.train()

    # acc, loss = validate(forget_loader, model, criterion, args)
    # print("Forget Accuracy: ", acc)
    # print("Forget Loss: ", loss)

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.to(device)
        target = target.to(device)
        if epoch < args.unlearn_epochs - args.no_l1_epochs:
            current_alpha = args.alpha * (
                1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
            )
        else:
            current_alpha = 0
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)
        if with_l1:
            loss += current_alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        #Get the code to work
        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        train_loss.update(loss.item(), image.size(0))
        train_accuracy.update(prec1.item(), image.size(0))

        # Sample from the forget loader
        image, target = next(iter(forget_loader))
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        loss = criterion(output, target)
        prec1 = utils.accuracy(output.data, target)[0]
        forget_losses.update(loss.item(), image.size(0))
        forget_accuracies.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=train_loss, top1=train_accuracy
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=train_accuracy))
    print("forget_accuracy {top1.avg:.3f}".format(top1=forget_accuracies))

    return train_accuracy.avg, train_loss.avg, forget_accuracies.avg, forget_losses.avg

def FT(data_loaders, model, criterion, optimizer, epoch, args, mask=None, wandb_run=None):
    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed

    for epoch in range(args.unlearn_epochs):
        iterr_acc, iterr_loss, iterf_acc, iterf_loss = FT_iter(data_loaders, model, criterion, optimizer, epoch, args, None)

        if wandb_run is not None:
            wandb_run.log({
                "FT In Process Forget Loss": iterf_loss,
                "FT In Process Forget Accuracy": iterf_acc,
                "FT In Process Retain Loss": iterr_loss,
                "FT In Process Retain Accuracy": iterr_acc,
            })

        r_accuracy.update(iterr_acc)
        r_loss.update(iterr_loss)
        f_accuracy.update(iterf_acc)
        f_loss.update(iterf_loss)

        scheduler.step()

    evaluations = {
        "train": {"loss": r_loss, "accuracy": r_accuracy},
        "forget": {"loss": f_loss, "accuracy": f_accuracy},
        "retain": {"loss": r_loss, "accuracy": r_accuracy},
        "epochs": args.unlearn_epochs,
    }

    return evaluations
