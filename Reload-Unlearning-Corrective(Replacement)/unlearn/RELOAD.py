import math
import sys
import time

import torch
import wandb
from torch import nn
import utils

from trainer import validate

sys.path.append(".")


def RELOAD(data_loaders, model, criterion, optimizer, epoch, args, mask=None, wandb_run=None):

    train_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    print(len(train_loader))

    train_loss = utils.AverageMeter()
    train_accuracy = utils.AverageMeter()

    r_loss = utils.AverageMeter()
    r_accuracy = utils.AverageMeter()

    f_loss = utils.AverageMeter()
    f_accuracy = utils.AverageMeter()

    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    # RELOAD is expected to checkpoint about halfway through the retraining process, and so
    # the learning rate scheduler should not start at the beginning. Let's take 40% of the retraining
    # epochs and use that as the starting point for the learning rate scheduler.

    # Additionally, we're going to decrease the distance between the different milestones as well
    # because RELOAD is regaining performance/repairing much faster than the original training process.

    checkpoint_epoch = 4 * args.epochs // 10

    # Subtract checkpoint_epoch from every element in decreasing_lr
    decreasing_scalar = 2 # Setting decreasing_scalar to 2
    denom = decreasing_scalar * len(decreasing_lr) if decreasing_scalar * len(decreasing_lr) > 0 else 3
    decreasing_lr = [decreasing_lr[i] - (checkpoint_epoch * (1 + (i / denom))) for i in range(len(decreasing_lr))]

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed

    # import trainer
    # trainer.validate(data_loaders["retain"], model, criterion, args)

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    f_val_acc, f_val_loss = validate(forget_loader, model, criterion, args)

    wandb_run.log({
        "RELOAD In Process Forget Loss": f_val_loss,
        "RELOAD In Process Forget Accuracy": f_val_acc,
    })

    # switch to train mode
    model.train()

    if args.ga_lr > 0.0: # and args.masking_mode != "obs":
        # Normalise the gradients
        # Iterate through forget_gradients

        gradients = args.forget_gradients

        for name, param in model.named_parameters():
            param.data.copy_(param + args.ga_lr * gradients[name])

    # Save checkpoint
    filepath = f"./trained_models/{args.dataset}_{args.arch}_prime.pt"
    torch.save(model.state_dict(), filepath)
    torch.save(gradients, f"./trained_models/{args.dataset}_{args.arch}_fga.pt")

    f_val_acc, f_val_loss = validate(forget_loader, model, criterion, args)

    wandb_run.log({
        "RELOAD In Process Forget Loss": f_val_loss,
        "RELOAD In Process Forget Accuracy": f_val_acc,
    })

    if mask:
        for name, param in model.named_parameters():
            mask[name] = mask[name].to(device)
            inverted_mask = torch.ones_like(mask[name]) - mask[name]

            # Added some weight initialization methods
            # Zero, Uniform, Normal, Kaiming, Ian, Xavier, etc.
            try:
                zerod = inverted_mask * param
            except:
                if len(param.shape) == 1:
                    param = param.unsqueeze(-1)
                zerod = inverted_mask * param
            # if len(zerod.shape) == 1:
                # zerod = zerod.unsqueeze(-1)
            zeros_like = torch.zeros_like(zerod)
            # Ensure that zeros like has 2 dimensions
            if len(zeros_like.shape) == 1:
                zeros_like = zeros_like.unsqueeze(-1)
            init = zeros_like
            if args.init == "Zero":
                init = init
            elif args.init == "Uniform":
                init = torch.nn.init.uniform_(zeros_like, -1, 1)
            elif args.init == "Normal":
                init = torch.nn.init.normal_(zeros_like, 0, 1)
            elif args.init == "Xavier_Uniform":
                init = torch.nn.init.xavier_uniform_(zeros_like)
            elif args.init == "Xavier_Normal":
                init = torch.nn.init.xavier_normal_(zeros_like)
            elif args.init == "Kaiming_Uniform":
                init = torch.nn.init.kaiming_uniform_(zeros_like)
            elif args.init == "Kaiming_Normal":
                init = torch.nn.init.kaiming_normal_(zeros_like)
            elif args.init == "Ian_Uniform":
                if len(inverted_mask.shape) == 1:
                    inverted_mask = inverted_mask.unsqueeze(0)
                # Calculate the number of 1s in the inverted mask as the unchanged weights
                # Ensuring non-zero fan-in
                nonlinearity = "leaky_relu"
                fan_ins = inverted_mask.sum(dim=1).clamp(min=1)
                # Note a parameter is only necessary for leaky_relu non-linearity
                gain = torch.nn.init.calculate_gain(nonlinearity, 0)
                # Compute standard deviation for weights
                std = gain / torch.sqrt(fan_ins)
                # Compute bounds for uniform distribution
                bounds = std * math.sqrt(3.0)
                # Apply uniform initialization row-wise to handle varying bounds
                init = torch.nn.init.uniform_(zeros_like, -bounds.mean().item(), bounds.mean().item())
            elif args.init == "Ian_Normal":
                if len(inverted_mask.shape) == 1:
                    inverted_mask = inverted_mask.unsqueeze(0)
                # Calculate the number of 1s in the inverted mask as the unchanged weights
                nonlinearity = "leaky_relu"
                gain = torch.nn.init.calculate_gain(nonlinearity, 0)
                num_ones = inverted_mask.sum(dim=1).clamp(min=1)
                std = gain / torch.sqrt(num_ones)
                init = torch.nn.init.normal_(zeros_like, 0, std.mean().item())
            elif args.init == "Avi_Normal" or args.init == "Avi_Uniform":
                # Initialize the weights proportional to their importance
                forget_gradients = args.forget_gradients
                forget_gradients = forget_gradients[name].to(device)
                forget_gradients = torch.abs(forget_gradients)
                # Find the minimum gradient
                min_grad = torch.min(forget_gradients)
                max_grad = torch.max(forget_gradients)
                std_dev = max_grad - min_grad + 1e-10
                distrib = torch.distributions.Normal(min_grad, std_dev)
                # forget gradient and init are of the same shape and size, the position at init is the position in forget_graidnet under te distribution
                init = distrib.cdf(forget_gradients) - (distrib.cdf(min_grad)) * torch.ones_like(forget_gradients)
                init = init * param

            if init.shape != mask[name].shape:
                init = init.reshape(mask[name].shape)
            new_param = zerod + init * mask[name]
            # Make sure the new_param is the same shape as the original param
            if new_param.shape != param.shape:
                new_param = new_param.squeeze(0)
            try:
                param.data.copy_(new_param)
            except:
                print(f"Mask Shape: {mask[name].shape}")
                print(f"Param Shape: {param.shape}")
                print(f"Init Method: {args.init}")
                print(f"New Param Shape: {new_param.shape}")
                print(f"Zerod Shape: {zerod.shape}")
                print(f"Init Tensor Shape: {init.shape}")
                print(f"Init * Mask[name] Shape: {(init * mask[name]).shape}")
                wandb_run.finish()
                exit() # Kill the process

            # Requires grad is true by default

    # Save checkpoint
    filepath = f"./trained_models/{args.dataset}_{args.arch}_reset.pt"
    torch.save(model.state_dict(), filepath)
    torch.save(mask, f"./trained_models/{args.dataset}_{args.arch}_mask.pt")

    f_val_acc, f_val_loss = validate(forget_loader, model, criterion, args)

    wandb_run.log({
        "RELOAD In Process Forget Loss": f_val_loss,
        "RELOAD In Process Forget Accuracy": f_val_acc,
    })

    epoch_counter = 0

    while epoch_counter < args.unlearn_epochs:
        print("Starting epoch: ", epoch_counter)
        # utils.plot_decision_boundary(model, f"ZERO_epoch_{epoch_counter}", data_loaders, "forget", device, args, True)
        train_loss = utils.AverageMeter()
        train_accuracy = utils.AverageMeter()
        forget_losses = utils.AverageMeter()
        forget_accs = utils.AverageMeter()
        start = time.time()
        model.train()
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            train_loss.update(loss.item(), image.size(0))
            train_accuracy.update(prec1.item(), image.size(0))

            with torch.no_grad():
                # Randomly sample from the forget loader
                image, target = next(iter(forget_loader))
                image = image.to(device)
                target = target.to(device)

                # compute output
                output_clean = model(image)
                loss = criterion(output_clean, target)

                prec1 = utils.accuracy(output_clean.data, target)[0]

                forget_losses.update(loss.item(), image.size(0))
                forget_accs.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print("RELOAD\n"
                      "Epoch: [{0}][{1}/{2}]\t"
                      "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                      "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                      "Time {3:.2f}".format(
                    epoch_counter, i, len(train_loader), end - start, loss=train_loss, top1=train_accuracy)
                )
                start = time.time()

        f_loss.update(forget_losses.avg, 1)
        f_accuracy.update(forget_accs.avg, 1)

        # wandb_run.log({
        #     "RELOAD In Process Forget Loss": forget_losses.avg,
        #     "RELOAD In Process Forget Accuracy": forget_accs.avg,
        # })

        f_val_acc, f_val_loss = validate(forget_loader, model, criterion, args)

        wandb_run.log({
            "RELOAD In Process Forget Loss": f_val_loss,
            "RELOAD In Process Forget Accuracy": f_val_acc,
        })

        r_loss.update(train_loss.avg, 1)
        r_accuracy.update(train_accuracy.avg, 1)

        wandb_run.log({
            "RELOAD In Process Retain Loss": train_loss.avg,
            "RELOAD In Process Retain Accuracy": train_accuracy.avg,
        })

        scheduler.step()

        epoch_counter += 1

        print("train_accuracy {top1.avg:.3f}".format(top1=train_accuracy))
        print("train_loss {loss.avg:.4f}".format(loss=train_loss))

        # Validate on the retain and forget loaders and log it
        # r_val_acc, r_val_loss = validate(train_loader, model, criterion, args)
        # f_val_acc, f_val_loss = validate(forget_loader, model, criterion, args)

        # wandb.log({
        #     "RELOAD In Process Retain Validation Accuracy": r_val_acc,
        #     "RELOAD In Process Retain Validation Loss": r_val_loss,
        #     "RELOAD In Process Forget Validation Accuracy": f_val_acc,
        #     "RELOAD In Process Forget Validation Loss": f_val_loss
        # })

    evaluations = {
            "train": {"loss": r_loss, "accuracy": r_accuracy},
            "forget": {"loss": f_loss, "accuracy": f_accuracy},
            "retain": {"loss": r_loss, "accuracy": r_accuracy},
            "epochs": epoch_counter,
        }

    return evaluations
