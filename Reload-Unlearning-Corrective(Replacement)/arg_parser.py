import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch RELOAD Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="./data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=2,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default="./trained_models",
        type=str,
    )
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )

    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget. We don't actually set this but use it as a cache in the code.",
    )
    parser.add_argument(
        "--prop_forget",
        type=float,
        default=0.1,
        help="Proportion of data to forget",
    )
    parser.add_argument(
        "--forget_class", type=int, default=-1, help="Specific class to forget when unlearning/remedial learning a subset from a specific class.",
    )
    parser.add_argument(
        "--class_to_replace", type=int, default=-1, help="Specific class to forget the whole class"
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )

    parser.add_argument("--alpha", default=0, type=float, help="unlearn noise")

    parser.add_argument("--gamma", default=1.0, type=float, help="original criteria weight. Used as caches for SCRUB.")
    parser.add_argument("--beta", default=0.0, type=float, help="kl criteria weight. Used as caches for SCRUB.")

    parser.add_argument("--path", default=None, type=str, help="mask matrix")
    parser.add_argument('--num_iter', default=None, type=int, help='the number of iteration')

    parser.add_argument("--threshold", default=0.9, type=float, help="1 - the actual threshold")

    parser.add_argument("--ga_lr", default=0.0, type=float, help="learning rate for initial ga step before ZERO")

    parser.add_argument("--forget_gradients", default=None, type=torch.tensor, help="forget gradients. Cache.")

    parser.add_argument("--masking_mode", default="std", type=str, help="Method for masking")

    parser.add_argument("--run_id", default=None, type=str, help="run id. cache for wandb.")

    parser.add_argument("--sweep_dataset", default=None, type=str, help="sweep dataset. override for dataset.")

    parser.add_argument("--init", default="Zero", type=str, help="weight initialization method for RELOAD")

    parser.add_argument("--task", default="remove", type=str, help="task in [remove, replace]")

    parser.add_argument("--replacement_type", default="noisy", type=str, help="replacement type in [noisy, poisoning, incorrect_label, targeted_label_flip")

    parser.add_argument("--noise_type", default="gaussian", type=str, help="noise type. [gaussian, poisson, salt, pepper s&p, speckle]")

    parser.add_argument("--noise_level", default=0.4, type=str, help="amount of noise inserted")

    parser.add_argument("--device", default=None, type=str, help="device. cache")

    parser.add_argument("--backdoor_dir", type=str, help="where the backdoor details are stored")

    parser.add_argument("--backdoor_type", type=str, default="cross_pattern", help="type of backdoor pattern to insert in [cross_pattern, distributed_pattern, feature_pattern]")

    return parser.parse_args()