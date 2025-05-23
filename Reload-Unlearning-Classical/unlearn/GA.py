import sys
import time

import torch
import utils
import wandb

from .impl import iterative_unlearn

sys.path.append(".")


def GA(data_loaders, model, criterion, optimizer, epoch, args, mask=None, wandb_run=None):
    train_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    print(len(train_loader))

    train_loss = utils.AverageMeter()
    train_accuracy = utils.AverageMeter()

    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        1e-3, #It was args.unlearn_lr before that, then 1e-4
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # switch to train mode
    model.train()

    start = time.time()

    for epoch in range(args.unlearn_epochs):
        train_loss = utils.AverageMeter()
        train_accuracy = utils.AverageMeter()
        retain_losses = utils.AverageMeter()
        retain_accs = utils.AverageMeter()
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
                        # print(mask[name])

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            train_loss.update(loss.item(), image.size(0))
            train_accuracy.update(prec1.item(), image.size(0))

            # Sample from the retain loader
            image, target = next(iter(retain_loader))
            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target).float()

            prec1 = utils.accuracy(output_clean.data, target)[0]

            retain_losses.update(loss.item(), image.size(0))
            retain_accs.update(prec1.item(), image.size(0))

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

        r_loss.update(retain_losses.avg)
        r_accuracy.update(retain_accs.avg)

        wandb_run.log({
            "GA In Process Retain Loss": retain_losses.avg,
            "GA In Process Retain Accuracy": retain_accs.avg,
        })

        f_loss.update(train_loss.avg)
        f_accuracy.update(train_accuracy.avg)

        wandb_run.log({
            "GA In Process Forget Loss": train_loss.avg,
            "GA In Process Forget Accuracy": train_accuracy.avg,
        })

        scheduler.step()

        print("train_accuracy {top1.avg:.3f}".format(top1=train_accuracy))

    evaluations = {
        "train": {"loss": r_loss, "accuracy": r_accuracy},
        "forget": {"loss": f_loss, "accuracy": f_accuracy},
        "retain": {"loss": r_loss, "accuracy": r_accuracy},
        "epochs": args.unlearn_epochs,
    }

    return evaluations

def GAR(data_loaders, model, criterion, optimizer, epoch, args, mask=None, wandb_run=None):
    old_loader = data_loaders["full"]
    new_loader = data_loaders["retain"]

    train_loss = utils.AverageMeter()
    train_accuracy = utils.AverageMeter()

    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        1e-3,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # switch to train mode
    model.train()

    start = time.time()

    for epoch in range(5):
        train_loss = utils.AverageMeter()
        train_accuracy = utils.AverageMeter()
        retain_losses = utils.AverageMeter()
        retain_accs = utils.AverageMeter()
        for i, (image, target) in enumerate(old_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(old_loader), args=args
                )

            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
                        # print(mask[name])

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            train_loss.update(loss.item(), image.size(0))
            train_accuracy.update(prec1.item(), image.size(0))

            # Sample from the retain loader
            image, target = next(iter(new_loader))
            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target).float()

            prec1 = utils.accuracy(output_clean.data, target)[0]

            retain_losses.update(loss.item(), image.size(0))
            retain_accs.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(old_loader), end - start, loss=train_loss, top1=train_accuracy
                    )
                )
                start = time.time()

        r_loss.update(retain_losses.avg)
        r_accuracy.update(retain_accs.avg)

        wandb_run.log({
            "GA In Process Retain Loss": retain_losses.avg,
            "GA In Process Retain Accuracy": retain_accs.avg,
        })

        f_loss.update(train_loss.avg)
        f_accuracy.update(train_accuracy.avg)

        wandb_run.log({
            "GA In Process Forget Loss": train_loss.avg,
            "GA In Process Forget Accuracy": train_accuracy.avg,
        })

        scheduler.step()

        print("train_accuracy {top1.avg:.3f}".format(top1=train_accuracy))

    for epoch in range(args.unlearn_epochs):
        train_loss = utils.AverageMeter()
        train_accuracy = utils.AverageMeter()
        retain_losses = utils.AverageMeter()
        retain_accs = utils.AverageMeter()
        for i, (image, target) in enumerate(new_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(new_loader), args=args
                )

            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
                        # print(mask[name])

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            retain_losses.update(loss.item(), image.size(0))
            retain_accs.update(prec1.item(), image.size(0))

            # Sample from the retain loader
            image, target = next(iter(old_loader))
            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target).float()

            prec1 = utils.accuracy(output_clean.data, target)[0]

            train_loss.update(loss.item(), image.size(0))
            train_accuracy.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(old_loader), end - start, loss=train_loss, top1=train_accuracy
                    )
                )
                start = time.time()

        r_loss.update(retain_losses.avg)
        r_accuracy.update(retain_accs.avg)

        wandb_run.log({
            "GA In Process Retain Loss": retain_losses.avg,
            "GA In Process Retain Accuracy": retain_accs.avg,
        })

        f_loss.update(train_loss.avg)
        f_accuracy.update(train_accuracy.avg)

        wandb_run.log({
            "GA In Process Forget Loss": train_loss.avg,
            "GA In Process Forget Accuracy": train_accuracy.avg,
        })

        scheduler.step()

        print("train_accuracy {top1.avg:.3f}".format(top1=train_accuracy))

    evaluations = {
        "train": {"loss": r_loss, "accuracy": r_accuracy},
        "forget": {"loss": f_loss, "accuracy": f_accuracy},
        "retain": {"loss": r_loss, "accuracy": r_accuracy},
        "epochs": args.unlearn_epochs,
    }

    return evaluations

def GRDA(data_loaders, model, criterion, optimizer, epoch, args, mask=None, wandb_run=None):
    old_loader = data_loaders["full"]
    new_loader = data_loaders["retain"]

    train_loss = utils.AverageMeter()
    train_accuracy = utils.AverageMeter()

    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        1e-3,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # switch to train mode
    model.train()

    start = time.time()

    for epoch in range(args.unlearn_epochs):
        train_loss = utils.AverageMeter()
        train_accuracy = utils.AverageMeter()
        retain_losses = utils.AverageMeter()
        retain_accs = utils.AverageMeter()
        for i, (image, target) in enumerate(old_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(old_loader), args=args
                )

            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            oloss = -criterion(output_clean, target)

            optimizer.zero_grad()
            oloss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
                        # print(mask[name])

            output = output_clean.float()
            oloss = oloss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            train_loss.update(oloss.item(), image.size(0))
            train_accuracy.update(prec1.item(), image.size(0))

            # Sample from the retain loader
            image, target = next(iter(new_loader))
            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            nloss = criterion(output_clean, target)

            nloss.backward()
            nloss = nloss.float()

            prec1 = utils.accuracy(output_clean.data, target)[0]

            retain_losses.update(nloss.item(), image.size(0))
            retain_accs.update(prec1.item(), image.size(0))

            optimizer.step()

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(old_loader), end - start, loss=train_loss, top1=train_accuracy
                    )
                )
                start = time.time()

        r_loss.update(retain_losses.avg)
        r_accuracy.update(retain_accs.avg)

        wandb_run.log({
            "GA In Process Retain Loss": retain_losses.avg,
            "GA In Process Retain Accuracy": retain_accs.avg,
        })

        f_loss.update(train_loss.avg)
        f_accuracy.update(train_accuracy.avg)

        wandb_run.log({
            "GA In Process Forget Loss": train_loss.avg,
            "GA In Process Forget Accuracy": train_accuracy.avg,
        })

        scheduler.step()

        print("train_accuracy {top1.avg:.3f}".format(top1=train_accuracy))

    evaluations = {
        "train": {"loss": r_loss, "accuracy": r_accuracy},
        "forget": {"loss": f_loss, "accuracy": f_accuracy},
        "retain": {"loss": r_loss, "accuracy": r_accuracy},
        "epochs": args.unlearn_epochs,
    }

    return evaluations