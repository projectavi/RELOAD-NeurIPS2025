"""
    setup model and datasets
"""

import random
# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import time

import numpy as np
from sklearn import linear_model, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import random
from torch import nn
from torchvision import transforms

from dataset import *
from transform_dataset import *

from models import *

__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
]


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def save_checkpoint(
        state, is_SA_best, save_path, tag=0, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(tag) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath, os.path.join(save_path, str(tag) + "model_SA_best.pth.tar")
        )


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []

    def update(self, val, n=1):
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = sum(self.vals) / len(self.vals)


def dataset_convert_to_train(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = train_transform
    dataset.train = False


def dataset_convert_to_test(dataset, args=None):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def get_dataset_size(args):
    if args.dataset == "cifar10":
        train_full_loader, val_loader, _ = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        dataset_size = len(train_full_loader.dataset)
    elif args.dataset == "svhn":
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        dataset_size = len(train_full_loader.dataset)
    elif args.dataset == "cifar100":
        train_full_loader, val_loader, _ = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        dataset_size = len(train_full_loader.dataset)
    return dataset_size, train_full_loader


def replace_dropout_with_identity(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            setattr(model, name, nn.Identity())
        else:
            replace_dropout_with_identity(module)
    return model


def replace_relu_with(model, type):
    if type == "identity":
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(model, name, nn.Identity())
            else:
                replace_relu_with(module, type)
    elif type == "leaky_relu":
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(model, name, nn.LeakyReLU())
            else:
                replace_relu_with(module, type)
    return model


def setup_model_dataset(args, gradient_mode=False):
    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_full_loader, val_loader, _ = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)

        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "2d_synthetic":
        classes = 2
        n_samples = 5000
        normalization = NormalizeByChannelMeanStd(mean=[0.5, 0.5], std=[0.5, 0.5])
        train_full_loader, val_loader, _ = synthetic_dataloaders(
            batch_size=args.batch_size, num_workers=args.workers, seed=args.seed, n_classes=classes,
            n_samples=n_samples, n_features=2
        )
        marked_loader, _, test_loader = synthetic_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            n_classes=classes,
            n_samples=n_samples, n_features=2
        )
        model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)

        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "svhn":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
        )
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = svhn_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
        )
        model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_full_loader, val_loader, _ = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )
        model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "cifar100_no_val":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    elif args.dataset == "cifar10_no_val":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
    elif args.dataset == "mnist":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.1307], std=[0.3081]
        )
        train_set_loader, val_loader, test_loader = mnist_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = mnist_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )
    elif args.dataset == "usps":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.2469], std=[0.2811]
        )
        train_set_loader, val_loader, test_loader = usps_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = usps_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )
    elif args.dataset == "CensusIncoms":
        classes = 2
        train_set_loader, val_loader, test_loader = census_income_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = census_income_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
        )
    else:
        raise ValueError("Dataset not supprot yet !")
    # import pdb;pdb.set_trace()

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    print(model)

    return model, train_set_loader, val_loader, test_loader, marked_loader


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open("stop_{}.sh".format(dir), "w")
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, "w")
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
    )


def get_unlearn_loader(marked_loader, args):
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(
        forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(
        retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    print("datasets length: ", len(forget_dataset), len(retain_dataset))
    return forget_loader, retain_loader


def display_image(tensor):
    import matplotlib.pyplot as plt
    img = tensor.astype(np.uint8)
    plt.imshow(img)
    plt.show()

def get_poisoned_loader(poison_loader, unpoison_loader, test_loader, poison_func, args):
    poison_dataset = copy.deepcopy(poison_loader.dataset)
    poison_test_dataset = copy.deepcopy(test_loader.dataset)

    poison_dataset.data, poison_dataset.targets = poison_func(
        poison_dataset.data, poison_dataset.targets
    )
    poison_test_dataset.data, poison_test_dataset.targets = poison_func(
        poison_test_dataset.data, poison_test_dataset.targets
    )

    full_dataset = torch.utils.data.ConcatDataset(
        [unpoison_loader.dataset, poison_dataset]
    )

    poisoned_loader = get_loader_from_dataset(
        poison_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    poisoned_full_loader = get_loader_from_dataset(
        full_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    poisoned_test_loader = get_loader_from_dataset(
        poison_test_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )

    return poisoned_loader, unpoison_loader, poisoned_full_loader, poisoned_test_loader


def multi_func(tup, model_trained=None):
    method, args, forget_gradients, wandb_run = tup
    temp_args = copy.deepcopy(args)
    temp_args.unlearn = method[0]
    device = temp_args.device
    temp_args.device = method[2] if method[2] is not None else device
    dev = torch.device(temp_args.device)
    # Send everything in forget_gradients to the device
    for key in forget_gradients.keys():
        forget_gradients[key] = forget_gradients[key].to(dev)
    temp_args.forget_gradients = forget_gradients
    return method[0], method[1](temp_args, wandb_run, model_trained)

def multi_func_slurm(tup, model_trained=None):
    method, args, forget_gradients, wandb_run = tup
    temp_args = copy.deepcopy(args)
    temp_args.unlearn = method[0]
    device = temp_args.device
    temp_args.device = method[2] if method[2] is not None else device
    dev = torch.device(temp_args.device)
    # Send everything in forget_gradients to the device
    for key in forget_gradients.keys():
        forget_gradients[key] = forget_gradients[key].to(dev)
    temp_args.forget_gradients = forget_gradients


    # SLURM BATCH METHOD[1](temp_args, wandb_run, model_trained) - arguments


def compute_losses(model, loader, device):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def run_mia(model, device, test_loader, target_loader):
    target_losses = compute_losses(model, target_loader, device)
    test_losses = compute_losses(model, test_loader, device)

    np.random.shuffle(target_losses)
    target_losses = target_losses[: len(test_losses)]

    samples_mia = np.concatenate((test_losses, target_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(target_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)

    return mia_scores


### All code below here has been adapted from https://github.com/meghdadk/SCRUB/blob/main/MIA_experiments.ipynb

def cm_score(estimator, X, y):
    y_pred = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)

    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[0][0]
    TN = cnf_matrix[1][1]

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = ((TP + TN) / (TP + FP + FN + TN)) if (TP + FP + FN + TN) != 0 else 0
    # print(f"FPR:{FPR:.2f}, FNR:{FNR:.2f}, FP{FP:.2f}, TN{TN:.2f}, TP{TP:.2f}, FN{FN:.2f}")
    return ACC


def evaluate_attack_model(sample_loss,
                          members,
                          n_splits=5,
                          random_state=None):
    """Computes the cross-validation score of a membership inference attack.
  Args:
    sample_loss : array_like of shape (n,).
      objective function evaluated on n samples.
    members : array_like of shape (n,),
      whether a sample was used for training.
    n_splits: int
      number of splits to use in the cross-validation.
    random_state: int, RandomState instance or None, default=None
      random state to use in cross-validation splitting.
  Returns:
    score : array_like of size (n_splits,)
  """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = LogisticRegression()
    cv = StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state)
    return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=cm_score)


def scrub_mia_baselines(model, baseline, t_loader, f_loader, args, plot=False, log_model=None):
    import matplotlib.pyplot as plt

    cls_ignore = args.class_to_replace

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.dataset == "svhn":
        fgt_target = f_loader.dataset.labels
        t_target = t_loader.dataset.labels
    else:
        fgt_target = f_loader.dataset.targets
        t_target = t_loader.dataset.targets

    fgt_cls = list(np.unique(fgt_target))
    print(fgt_cls)
    indices = [i in fgt_cls for i in t_target]
    print(len(indices))
    print(t_target.shape)
    print(t_loader.dataset.data.shape)
    t_loader.dataset.data = t_loader.dataset.data[indices]
    t_target = t_target[indices]

    print(len(t_loader.dataset))

    model = model.to(device)

    # cr = nn.NLLLoss(reduction="none")
    cr = nn.CrossEntropyLoss(reduction="none")
    test_losses = []
    forget_losses = []
    model.eval()
    # mult = 0.5 if args.lossfn == 'mse' else 1
    mult = 1
    dataloader = torch.utils.data.DataLoader(t_loader.dataset, batch_size=128, shuffle=False)

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        # if args.lossfn == 'mse':
        #     target = (2 * target - 1)
        #     target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data = data.view(data.shape[0], -1)
        output = model(data)

        output = output.float()

        # Set the output of the class to ignore to 0
        output[:, cls_ignore] = 0

        # Normalize the output
        output = output / output.sum(dim=1, keepdim=True)

        if baseline == "Platt":
            # Calibrate the output by putting them all through the logistic regression
            new_output = torch.zeros_like(output)
            for i in range(output.shape[1]):
                if i == cls_ignore:
                    continue  # Dont calibrate and leave at 0
                i_prob = output[:, i].view(-1, 1)
                new_output[:, i] = log_model(i_prob).view(-1)
            output = new_output

        # output[:, cls_ignore] = float('-inf')

        loss = mult * cr(output, target)
        test_losses = test_losses + list(loss.cpu().detach().numpy())
    del dataloader
    dataloader = torch.utils.data.DataLoader(f_loader.dataset, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        # if args.lossfn == 'mse':
        #     target = (2 * target - 1)
        #     target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data = data.view(data.shape[0], -1)
        output = model(data)

        output = output.float()

        # Set the output of the class to ignore to 0
        output[:, cls_ignore] = 0

        # Normalize the output
        output = output / output.sum(dim=1, keepdim=True)

        if baseline == "Platt":
            # Calibrate the output by putting them all through the logistic regression
            new_output = torch.zeros_like(output)
            for i in range(output.shape[1]):
                if i == cls_ignore:
                    continue  # Dont calibrate and leave at 0
                i_prob = output[:, i].view(-1, 1)
                new_output[:, i] = log_model(i_prob).view(-1)
            output = new_output

        # output[:, cls_ignore] = float('-inf')

        loss = mult * cr(output, target)
        
        forget_losses = forget_losses + list(loss.cpu().detach().numpy())
    del dataloader

    print(f"forget: {len(forget_losses)}, test: {len(test_losses)}")

    np.random.seed(args.seed)
    random.seed(args.seed)
    if len(forget_losses) > len(test_losses):
        forget_losses = list(random.sample(forget_losses, len(test_losses)))
    elif len(test_losses) > len(forget_losses):
        test_losses = list(random.sample(test_losses, len(forget_losses)))

    print(f"forget: {len(forget_losses)}, test: {len(test_losses)}")

    if plot:
        # sns.distplot(np.array(test_losses), kde=False, norm_hist=False, rug=False, label='test-loss', ax=plt)
        # sns.distplot(np.array(forget_losses), kde=False, norm_hist=False, rug=False, label='forget-loss', ax=plt)
        plt.legend(prop={'size': 14})
        plt.tick_params(labelsize=12)
        plt.title("loss histograms", size=18)
        plt.xlabel('loss values', size=14)
        plt.show()
        print(np.max(test_losses), np.min(test_losses))
        print(np.max(forget_losses), np.min(forget_losses))

    test_labels = [0] * len(test_losses)
    forget_labels = [1] * len(forget_losses)
    # print(f"forget: {len(forget_labels)}, test: {len(test_labels)}")
    features = np.array(test_losses + forget_losses).reshape(-1, 1)
    labels = np.array(test_labels + forget_labels).reshape(-1)
    # print(f"features: {features}")
    # print(f"labels: {labels}")
    features = np.clip(features, -100, 100)
    # print(f"features: {features}")
    # print(f"labels: {labels}")
    score = evaluate_attack_model(features, labels, n_splits=5, random_state=args.seed)

    return score

def scrub_mia(model, t_loader, f_loader, args, plot=False):
    import matplotlib.pyplot as plt
    # import seaborn as sns

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.dataset == "svhn":
        fgt_target = f_loader.dataset.labels
        t_target = t_loader.dataset.labels
    else:
        fgt_target = f_loader.dataset.targets
        t_target = t_loader.dataset.targets

    fgt_cls = list(np.unique(fgt_target))
    print(fgt_cls)
    indices = [i in fgt_cls for i in t_target]
    print(len(indices))
    print(t_target.shape)
    print(t_loader.dataset.data.shape)
    t_loader.dataset.data = t_loader.dataset.data[indices]
    t_target = t_target[indices]

    model = model.to(device)

    cr = nn.CrossEntropyLoss(reduction='none')
    test_losses = []
    forget_losses = []
    model.eval()
    # mult = 0.5 if args.lossfn == 'mse' else 1
    mult = 1
    dataloader = torch.utils.data.DataLoader(t_loader.dataset, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        # if args.lossfn == 'mse':
        #     target = (2 * target - 1)
        #     target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data = data.view(data.shape[0], -1)
        output = model(data)
        loss = mult * cr(output, target)
        test_losses = test_losses + list(loss.cpu().detach().numpy())
    del dataloader
    dataloader = torch.utils.data.DataLoader(f_loader.dataset, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        # if args.lossfn == 'mse':
        #     target = (2 * target - 1)
        #     target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data = data.view(data.shape[0], -1)
        output = model(data)
        loss = mult * cr(output, target)
        forget_losses = forget_losses + list(loss.cpu().detach().numpy())
    del dataloader

    np.random.seed(args.seed)
    random.seed(args.seed)
    if len(forget_losses) > len(test_losses):
        forget_losses = list(random.sample(forget_losses, len(test_losses)))
    elif len(test_losses) > len(forget_losses):
        test_losses = list(random.sample(test_losses, len(forget_losses)))

    if plot:
        # sns.distplot(np.array(test_losses), kde=False, norm_hist=False, rug=False, label='test-loss', ax=plt)
        # sns.distplot(np.array(forget_losses), kde=False, norm_hist=False, rug=False, label='forget-loss', ax=plt)
        plt.legend(prop={'size': 14})
        plt.tick_params(labelsize=12)
        plt.title("loss histograms", size=18)
        plt.xlabel('loss values', size=14)
        plt.show()
        print(np.max(test_losses), np.min(test_losses))
        print(np.max(forget_losses), np.min(forget_losses))

    test_labels = [0] * len(test_losses)
    forget_labels = [1] * len(forget_losses)
    features = np.array(test_losses + forget_losses).reshape(-1, 1)
    labels = np.array(test_labels + forget_labels).reshape(-1)
    features = np.clip(features, -100, 100)
    print(f"features: {features}")
    print(f"labels: {labels}")
    score = evaluate_attack_model(features, labels, n_splits=5, random_state=args.seed)

    return score


def plot_decision_boundary(model, model_name, loaders, datasets, device, args, save_fig=False):
    import matplotlib.pyplot as plt

    forget_loader = loaders["forget"]
    retain_loader = loaders["retain"]
    full_loader = loaders["full"]

    loaders = [forget_loader, retain_loader, full_loader]
    datasets = ["forget", "retain", "full"]

    X = full_loader.dataset.data
    y = full_loader.dataset.targets

    x_0_min = X[:, 0].min()
    x_0_max = X[:, 0].max()
    x_1_min = X[:, 1].min()
    x_1_max = X[:, 1].max()

    # Grid size
    h = 0.1

    x_0_g, x_1_g = np.meshgrid(np.arange(x_0_min, x_0_max, h), np.arange(x_1_min, x_1_max, h))

    # Convert x_g to tensor
    x_g = torch.tensor(np.c_[x_0_g.ravel(), x_1_g.ravel()], dtype=torch.float32).to(device)

    # Get predictions
    model.eval()

    with torch.no_grad():
        y_g = model(x_g).argmax(dim=1).cpu().numpy().reshape(x_0_g.shape)

    # Plot
    plt.figure(figsize=(30, 10))

    for j in range(len(loaders)):
        plt.subplot(1, 3, j + 1)
        plt.contourf(x_0_g, x_1_g, y_g, alpha=0.3, cmap='tab20')

        X = loaders[j].dataset.data
        y = loaders[j].dataset.targets

        for i in range(len(np.unique(y))):
            plt.scatter(X[y == i, 0], X[y == i, 1], label=i)

        plt.title(f'{model_name} Decision boundary for {datasets[j]} dataset of {args.dataset}')

    plt.legend()

    if save_fig:
        title = args.run_id
        if title is None:
            title = "test"
        os.makedirs("./logs/decision_boundaries", exist_ok=True)
        plt.savefig(f"./logs/decision_boundaries/{title}_{args.dataset}_{args.arch}_{model_name}.png")
    else:
        plt.show()

    plt.close()

def relabel_data(data_loader, removed_class, args=None, eval_mode=False):
    # The removed class is not in the data loader
    # For every class greater than the removed class, subtract 1
    # For every class less than the removed class, do nothing
    new_targets = data_loader.dataset.labels if args.dataset == "svhn" else data_loader.dataset.targets
    for i in range(len(new_targets)):
        if new_targets[i] > removed_class:
            new_targets[i] -= 1
        if new_targets[i] == removed_class and eval_mode == True:
            new_targets[i] = 9 # Set to max which wont be in the output of the new model
    if args.dataset == "svhn":
        data_loader.dataset.labels = new_targets
    else:
        data_loader.dataset.targets = new_targets
    return data_loader

def replace_data(unlearn_data_loaders, args, model=None):
    if args.replacement_type == "noisy":
        unlearn_data_loaders = noisy_replacement(unlearn_data_loaders, args=args)
    if args.replacement_type == "outdated":
        unlearn_data_loaders = outdated_replacement(unlearn_data_loaders, args=args)
    if args.replacement_type == "feature_mask":
        unlearn_data_loaders = feature_replacement(unlearn_data_loaders, args=args)
    if args.replacement_type == "incorrect_label":
        unlearn_data_loaders = incorrect_labelling(unlearn_data_loaders, args=args)
    if args.replacement_type == "feature_removal":
        unlearn_data_loaders = feature_removal(unlearn_data_loaders, args=args)
    if args.replacement_type == "feature_invariance":
        unlearn_data_loaders = feature_invariance(unlearn_data_loaders, args=args)
    if args.replacement_type == "plausibly_incorrect_label":
        unlearn_data_loaders = plausibly_incorrect_labels(unlearn_data_loaders, args=args, model=model)
    if args.replacement_type == "targeted_label_flip":
        unlearn_data_loaders = targeted_label_flip(unlearn_data_loaders, args=args)
    if args.replacement_type == "domain_adaptation":
        unlearn_data_loaders = domain_adaptation(unlearn_data_loaders, args=args)
    if args.replacement_type == "poisoning":
        unlearn_data_loaders = poison_data(unlearn_data_loaders, args=args)

    return unlearn_data_loaders