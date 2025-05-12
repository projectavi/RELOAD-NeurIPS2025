import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train, validate
from utils import *
from utils import NormalizeByChannelMeanStd

best_sa = 0


def main(best_file_name=None, train_loader_override=None, args=None, wandb_run=None):
    global best_sa

    if args is None:
        args = arg_parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    if wandb_run is not None:
        wandb_run.log({"seed": args.seed})

    # prepare dataset
    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    else:
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
        ) = setup_model_dataset(args)
    model.to(device)

    if train_loader_override is not None:
        train_loader = train_loader_override

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
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

    all_result = {}
    all_result["train_ta"] = []
    all_result["test_ta"] = []
    all_result["val_ta"] = []

    start_epoch = 0
    state = 0
    # start_state = 0

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(
            "Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        acc = train(train_loader, model, criterion, optimizer, epoch, args, None)

        # evaluate on validation set
        tacc, tloss = validate(val_loader, model, criterion, args)
        # # evaluate on test set
        # test_tacc = validate(test_loader, model, criterion, args)

        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)
        # all_result['test_ta'].append(test_tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint(
            {
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_SA_best=is_best_sa,
            save_path=args.save_dir,
            filename=args.dataset + "_" + args.arch + "_model_SA_best.pth.tar" if best_file_name is None else best_file_name,
        )
        print("one epoch duration:{}".format(time.time() - start_time))

    # plot training curve
    plt.plot(all_result["train_ta"], label="train_acc")
    plt.plot(all_result["val_ta"], label="val_acc")
    # plt.plot(all_result['test_ta'], label='test_acc')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, str(0) + "net_train.png"))
    plt.close()

    # report result
    # check_sparsity(model)
    print("Performance on the test data set")
    test_tacc, test_loss = validate(val_loader, model, criterion, args)
    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* best SA = {}, Epoch = {}".format(
                all_result["val_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
            )
        )


if __name__ == "__main__":
    main()