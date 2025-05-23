"""
This file will contain methods for applying different types of transformations to the dataset, for the purpose of testing selective data replacement.

The goal here will be to use these methods to create a pipeline where a dataset is produced and a subset of it is transformed in some way.

Then, that same subset is unlearned, and the original untransformed data is added to the retain dataset.

Each function takes in the full dataset and returns an ordered dict containing the different parts.
"""

import copy
import os
import pickle

import torch
import torch.nn as nn

import numpy
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from torchvision import transforms

from collections import OrderedDict

from skimage.util import random_noise
from tqdm import tqdm
import utils
import numpy as np

from skimage.color import rgb2gray

"""
Procedure setup can include this. THe methods will take in either an ordered dict or a single dataset. if it is a single dataset then this is the training regime.

If an ordered dict is passed in then we are on the unlearning stage, read from the file and replace the loaders appropriately.
"""


# Adapted from https://github.com/alewarne/MachineUnlearning/blob/main/Applications/Poisoning/poison/patterns.py
def create_backdoor_pattern(pattern_type, img_shape, args):
    # Parameters for cross_pattern
    center = False
    offset = 0
    cross_value = 0.5
    cross_size = 4

    # Parameters for distributed_pattern
    n_pixels = 10
    pixel_value = 0.5

    # Parameters for feature_pattern
    n_feat = 10
    pixel_value = 1.0

    # Parameters for noise pattern
    l_inf_norm = 0.1

    # All images for these experiments are 32 x 32
    rows, cols = 32, 32
    channels = 3

    if pattern_type == "cross_pattern":
        backdoor_pattern = np.zeros(img_shape)

        # Options include center and offset to demonstrate different places to inject the backdoor pattern
        if center:
            row_anchor = rows // 2
            col_anchor = cols // 2
        elif offset > 0:
            row_anchor = rows - offset
            col_anchor = cols - offset
        else:
            row_anchor = rows
            col_anchor = cols

        for i in range(cross_size + 1):
            if args.dataset != 'svhn':
                # moving from bottom right to top left
                backdoor_pattern[row_anchor - 1 - i, col_anchor - 1 - i, :] = cross_value
                # moving from bottom left to top right
                backdoor_pattern[row_anchor - 1 - i, col_anchor - 1 - cross_size + i, :] = cross_value
            else:
                # moving from bottom right to top left
                backdoor_pattern[:, row_anchor - 1 - i, col_anchor - 1 - i] = cross_value
                # moving from bottom left to top right
                backdoor_pattern[:, row_anchor - 1 - i, col_anchor - 1 - cross_size + i] = cross_value
    elif pattern_type == "distributed_pattern":
        backdoor_pattern = np.zeros(img_shape)
        np.random.seed(args.seed)
        bd_pixels = np.random.randint(low=0, high=rows, size=(n_pixels, 2))
        if args.dataset == "svhn":
            backdoor_pattern[:, bd_pixels[:, 0], bd_pixels[:, 1]] = pixel_value
        else:
            backdoor_pattern[bd_pixels[:, 0], bd_pixels[:, 1], :] = pixel_value
    elif pattern_type == "feature_pattern":
        np.random.seed(args.seed)
        backdoor_pattern = np.zeros(np.prod(img_shape))
        bd_feat = np.random.randint(low=0, high=backdoor_pattern.shape[0], size=n_feat)
        backdoor_pattern[bd_feat] = pixel_value
        backdoor_pattern = backdoor_pattern.reshape(img_shape)
    elif pattern_type == "noise_pattern":
        np.random.seed(args.seed)
        if args.dataset == "svhn":
            backdoor_pattern = np.random.uniform(low=0.0, high=l_inf_norm, size=(channels, rows, cols))
        else:
            backdoor_pattern = np.random.uniform(low=0.0, high=l_inf_norm, size=(rows, cols, channels))
    else:
        raise NotImplementedError("Pattern Type Not Found")

    return backdoor_pattern

def inject_backdoor_loader(eval_data_loader, args=None):

    new_loader = copy.deepcopy(eval_data_loader)

    with open(f"{args.backdoor_dir}/backdoor.pkl", 'rb') as f:
        pair = pickle.load(f)

    backdoor_pattern = pair["pattern"]

    # Add the backdoor pattern into the evaluation data and test its accuracy
    if args.backdoor_type in ["distributed_pattern", "feature_pattern"]:
        for i in range(len(new_loader.dataset.data)):
            np.copyto(new_loader.dataset.data[i], backdoor_pattern, where=backdoor_pattern > 0)
    else:
        new_loader.dataset.data += backdoor_pattern

    return new_loader


def poison_data(data, args=None):
    # Backdoor type will dictate the kind of backdoor/poison attack injected
    # Options are: cross_pattern, distributed_pattern, feature_pattern, noise_pattern

    import pickle

    if args.backdoor_type not in ["cross_pattern", "distributed_pattern", "feature_pattern", "noise_pattern"]:
        raise NotImplementedError

    if isinstance(data, OrderedDict):
        # Unlearning case. Here we poison the train and forget (old) set, and load the retain set as the pure dataset.

        train_loader = data['full']
        retain_loader = copy.deepcopy(train_loader)
        forget_loader = copy.deepcopy(data['forget'])

        # Load the backdoor pair
        if args.backdoor_type != "cross_pattern":
            with open(f"{args.backdoor_dir}/backdoor.pkl", 'rb') as f:
                pair = pickle.load(f)

            backdoor_pattern = pair["pattern"]
            assert (args.indexes_to_replace == pair['indices'])
        else:
            img_sample = train_loader.dataset.data[0]
            img_shape = img_sample.shape
            backdoor_pattern = create_backdoor_pattern(args.backdoor_type, img_shape, args)
            backdoor_pattern = (backdoor_pattern * 255).astype(np.uint8)

            pair = {"pattern": backdoor_pattern,
                "indices": args.indexes_to_replace}

            os.makedirs(args.backdoor_dir, exist_ok=True)
            with open(f"{args.backdoor_dir}/backdoor.pkl", 'wb') as f:
                pickle.dump(pair, f, protocol=pickle.HIGHEST_PROTOCOL)

        if args.backdoor_type in ["distributed_pattern", "feature_pattern"]:
            for i in args.indexes_to_replace:
                np.copyto(train_loader.dataset.data[i], backdoor_pattern, where=backdoor_pattern > 0)
            for i in range(len(forget_loader.dataset.data)):
                np.copyto(forget_loader.dataset.data[i], backdoor_pattern, where=backdoor_pattern > 0)
        else:
            train_loader.dataset.data[args.indexes_to_replace] += backdoor_pattern
            forget_loader.dataset.data += backdoor_pattern

        return OrderedDict(full=train_loader, retain=retain_loader, forget=forget_loader, val=data['val'],
                           test=data['test'])
    else:
        # Training case. Here we need to poison the data and dump the poisoning type

        train_loader = data

        img_sample = train_loader.dataset.data[0]
        img_shape = img_sample.shape

        backdoor_pattern = create_backdoor_pattern(args.backdoor_type, img_shape, args)
        backdoor_pattern = (backdoor_pattern * 255).astype(np.uint8)

        # from matplotlib import pyplot as plt
        # plt.imshow(backdoor_pattern, interpolation='nearest')
        # plt.show()

        # Inject the backdoor into the samples we are unlearning
        if args.backdoor_type in ["distributed_pattern", "feature_pattern"]:
            for i in args.indexes_to_replace:
                np.copyto(train_loader.dataset.data[i], backdoor_pattern, where=backdoor_pattern > 0)
        else:
            train_loader.dataset.data[args.indexes_to_replace] += backdoor_pattern

        # from matplotlib import pyplot as plt
        # plt.imshow(train_loader.dataset.data[args.indexes_to_replace[0]],
        #            interpolation='nearest')
        # plt.show()

        pair = {"pattern": backdoor_pattern,
                "indices": args.indexes_to_replace}

        os.makedirs(args.backdoor_dir, exist_ok=True)
        with open(f"{args.backdoor_dir}/backdoor.pkl", 'wb') as f:
            pickle.dump(pair, f, protocol=pickle.HIGHEST_PROTOCOL)

        return train_loader


def feature_invariance(data, args=None):
    seed = args.seed
    utils.setup_seed(seed)

    if isinstance(data, OrderedDict):
        import os

        job_id = os.environ.get("SLURM_JOB_ID")
        job_dir = f"/checkpoint/newatiaa/{job_id}"

        try:
            os.makedirs(job_dir, exist_ok=True)
        except:
            print("Dir Exists")

        train_loader = data["full"]
        forget_loader = copy.deepcopy(train_loader)

        data_name = f"/invar_data_{args.dataset}_{args.seed}.pt"
        label_name = f"/invar_label_{args.dataset}_{args.seed}.pt.pt"

        data_file = job_dir + data_name
        label_file = job_dir + label_name

        if not torch.cuda.is_available():
            os.makedirs("./checkpoint", exist_ok=True)

        checkpoints = os.listdir("/checkpoint/newatiaa") if torch.cuda.is_available() else os.listdir("./checkpoint")
        for job in checkpoints:
            if os.path.exists(os.path.join(job, data_name)) and os.path.exists(os.path.join(job, label_name)):
                source_data = os.path.join(job, data_name)
                source_label = os.path.join(job, label_name)

                import shutil
                shutil.copy(source_data, job_dir)
                shutil.copy(source_label, job_dir)

        if os.path.exists(data_file) and os.path.exists(label_file):
            retain_set = torch.load(data_file)
            retain_labels = torch.load(label_file)
        else:
            retain_set = copy.deepcopy(train_loader.dataset.data)
            retain_labels = copy.deepcopy(
                train_loader.dataset.targets if args.dataset != "svhn" else train_loader.dataset.labels)

            tr = transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5,
                                       contrast=0.5,
                                       saturation=0.5,
                                       hue=0.1)
            ], p=0.8)

            pipe = transforms.Compose([
                transforms.ToTensor(),
                tr,
            ])

            data_col = None
            label_col = np.array([])

            n = len(retain_set)

            with tqdm(total=n) as time:
                for i in range(n):
                    # Augment the image with colour distortion
                    img = retain_set[i]
                    # print(img.shape)
                    if args.dataset == "svhn":
                        img = np.moveaxis(img, 0, 2)
                        # print(img.shape)
                    newone = pipe(img).numpy()
                    # newtwo = pipe(img).numpy()

                    newone = (newone * 255).astype(np.uint8)
                    # newtwo = (newtwo * 255).astype(np.uint8)
                    # print(newone.shape)

                    # if args.dataset == "svhn":
                    # newone = np.moveaxis(newone, 2, 0)
                    # newtwo = np.moveaxis(newtwo, 2, 0)

                    data_col = np.expand_dims(newone, axis=0) if data_col is None else np.append(data_col,
                                                                                                 np.expand_dims(newone,
                                                                                                                axis=0),
                                                                                                 axis=0)
                    # data_col = np.append(data_col, newtwo, axis=0)

                    label_col = np.append(label_col, retain_labels[i])

                    if data_col.shape[0] > 2500:
                        print(retain_set.shape)
                        print(data_col.shape)
                        retain_set = np.concatenate((retain_set, data_col), axis=0)
                        retain_labels = np.concatenate((retain_labels, label_col), axis=0)
                        data_col = None
                        label_col = np.array([])
                        # break
                    # label_col = np.append(label_col, retain_labels[i])

                    time.update(1)

            try:
                retain_set = np.concatenate((retain_set, data_col), axis=0)
                retain_labels = np.concatenate((retain_labels, label_col), axis=0)
            except:
                print("All points added through batching")

            if torch.cuda.is_available():
                torch.save(retain_set, data_file)
                torch.save(retain_labels, label_file)

        retain_loader = copy.deepcopy(train_loader)

        retain_loader.dataset.data = retain_set
        if args.dataset == "svhn":
            retain_loader.dataset.labels = retain_labels
        else:
            retain_loader.dataset.targets = retain_labels

        return OrderedDict(full=train_loader, forget=forget_loader, retain=retain_loader, val=data["val"],
                           test=data["test"])
    else:
        return data  #train step nothing needs to be done


def feature_removal(data, args=None):
    # In this case there is indexes to replace, but the entire dataset needs to be grayscaled and returned.

    seed = args.seed

    if isinstance(data, OrderedDict):
        # This is an unlearning step

        train_loader = data["full"]
        forget_loader = copy.deepcopy(train_loader)

        retain_loader = copy.deepcopy(train_loader)
        for i in range(len(retain_loader.dataset.data)):
            channel = np.mean(retain_loader.dataset.data[i], axis=0)
            retain_loader.dataset.data[i] = np.stack((channel,) * 3, axis=0)

        d = OrderedDict(
            full=train_loader, val=data['val'], test=data["test"], forget=forget_loader, retain=retain_loader
        )

        return d
    else:
        return data


def plausibly_incorrect_labels(data, model, args=None):
    if args is None:
        raise ValueError("Args must be provided")

    seed = args.seed
    indices = args.indexes_to_replace
    device = args.device

    # Take the data and incorrectly label the forget set, so that in the replacement process it can be replaced with
    # correct labelling

    utils.setup_seed(seed)

    print("Messing with labels")

    if isinstance(data, OrderedDict):
        # In this case we are in the unlearning step
        # Thus, we want to incorrectly label the forget set and those bits in the train_full set
        # And return a copy of the train_full set in the place of the retain set

        train_loader = data["full"]
        forget_loader = data["forget"]

        retain_loader = copy.deepcopy(train_loader)

        if args.dataset == "svhn" or args.dataset == "cifar10":
            classes = 10
        elif args.dataset == "cifar100":
            classes = 100
        else:
            raise NotImplementedError("This dataset is not implemented")

        class_list = list(set([i for i in range(classes)]))

        # for i in indices:
        #     temp = copy.copy(class_list)
        #     label = train_loader.dataset.labels[i] if args.dataset == "svhn" else train_loader.dataset.targets[i]
        #     temp.remove(label)
        #
        #     output = nn.functional.softmax(model(train_loader.dataset.data[i]))
        #     _, topk_idx = torch.topk(output, k=3)
        #
        #     # Remove label from topk_idx if it's in there
        #     if label in topk_idx:
        #         topk_idx = topk_idx[topk_idx != label]
        #
        #     # Choose a random label from the topk_idx
        #     new_label = np.random.choice(topk_idx.cpu().numpy())
        #
        #     if args.dataset == "svhn":
        #         train_loader.dataset.labels[i] = new_label
        #     else:
        #         train_loader.dataset.targets[i] = new_label

        new_labels = torch.load(
            f"{args.data}/new_labels_{args.dataset}_{args.seed}_{args.arch}_{args.num_indexes_to_replace}.pt")

        if args.dataset == "svhn":
            train_loader.dataset.labels[indices] = new_labels[indices]
            forget_loader.dataset.labels = train_loader.dataset.labels[indices]
        else:
            train_loader.dataset.targets[indices] = new_labels[indices]
            forget_loader.dataset.targets = train_loader.dataset.targets[indices]

        return_dict = OrderedDict(
            full=train_loader, retain=retain_loader, forget=forget_loader, val=data["val"], test=data["test"])

        return return_dict
    else:
        # In this instance data is the train loader and we are in the training phase
        # This function just needs to replace with random labels.

        train_loader = data

        if args.dataset == "svhn" or args.dataset == "cifar10" or args.dataset == "mnist" or args.dataset == "usps":
            classes = 10
        elif args.dataset == "cifar100":
            classes = 100
        else:
            raise NotImplementedError("This dataset is not implemented")

        class_list = list(set([i for i in range(classes)]))

        forget_loader = copy.deepcopy(train_loader)
        forget_loader.dataset.data = forget_loader.dataset.data[indices]
        if args.dataset == "svhn":
            forget_loader.dataset.labels = forget_loader.dataset.labels[indices]
        else:
            forget_loader.dataset.targets = forget_loader.dataset.targets[indices]

        forget_loader = DataLoader(forget_loader.dataset, batch_size=1)

        i = 0
        for idx, (img, label) in enumerate(forget_loader):
            temp = copy.copy(class_list)
            temp.remove(label)

            img = img.to(device)
            label = label.to(device)

            output = model(img)
            output = nn.functional.softmax(output)
            _, topk_idx = torch.topk(output, k=2)

            # Remove label from topk_idx if it's in there
            if label in topk_idx:
                topk_idx = topk_idx[topk_idx != label]

            # Choose a random label from the topk_idx
            print(f"Top K: {topk_idx}")
            try:
                new_label = topk_idx.squeeze()[0]
            except:
                new_label = topk_idx.squeeze().item()
            print(f"New Label{new_label}")

            if args.dataset == "svhn":
                train_loader.dataset.labels[i] = new_label
            else:
                train_loader.dataset.targets[i] = new_label

            i += 1

        # Save the new labels
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(train_loader.dataset.labels if args.dataset == "svhn" else train_loader.dataset.targets,
                   f"{args.data}/new_labels_{args.dataset}_{args.seed}_{args.arch}_{args.num_indexes_to_replace}.pt")

        return train_loader


def targeted_label_flip(data, args=None):
    if args is None:
        raise ValueError("Args must be provided")

    seed = args.seed
    indices = args.indexes_to_replace

    # Take the data and incorrectly label the forget set, so that in the replacement process it can be replaced with
    # correct labelling

    utils.setup_seed(seed)

    print("Messing with labels")

    flip_mappings = {
        "cifar10": {
            0: 2,
            2: 0,
            3: 5,
            5: 3,
            1: 9,
            9: 1,
        },
        "svhn": {
            0: 6,
            1: 7,
            2: 5,
            3: 8,
            4: 9,
            5: 3,
            6: 0,
            7: 1,
            8: 3,
            9: 4
        },
        "cifar100": {
            0: 1,
            50: 53,
            75: 77,
            60: 61,
            17: 18,
            24: 26,
            37: 38,
            82: 83,
            56: 58,
            92: 93
        }
    }

    flip_dict = flip_mappings[args.dataset]

    log_file = f"{args.backdoor_dir}/label.pkl"
    if os.path.exists(log_file):
        file = open(log_file, 'rb')
        class_to_flip = pickle.load(file)
    else:
        class_to_flip = np.random.choice(list(flip_dict.keys()))
        pickle.dump(class_to_flip, open(log_file, 'wb'))

    target_class = flip_dict[class_to_flip]

    if isinstance(data, OrderedDict):
        # In this case we are in the unlearning step
        # Thus, we want to incorrectly label the forget set and those bits in the train_full set
        # And return a copy of the train_full set in the place of the retain set

        train_loader = data["full"]
        forget_loader = data["forget"]

        retain_loader = copy.deepcopy(train_loader)

        indices = []

        for i in range(len(train_loader.dataset.data)):
            label = train_loader.dataset.labels[i] if args.dataset == "svhn" else train_loader.dataset.targets[i]

            if label == class_to_flip:
                if args.dataset == "svhn":
                    train_loader.dataset.labels[i] = target_class
                else:
                    train_loader.dataset.targets[i] = target_class

                indices.append(i)

        if args.dataset == "svhn":
            forget_loader.dataset.labels = train_loader.dataset.labels[indices]
        else:
            forget_loader.dataset.targets = train_loader.dataset.targets[indices]

        forget_loader.dataset.data = train_loader.dataset.data[indices]

        return_dict = OrderedDict(
            full=train_loader, retain=retain_loader, forget=forget_loader, val=data["val"], test=data["test"])

        return return_dict
    else:
        # In this instance data is the train loader and we are in the training phase
        # This function just needs to replace with random labels.

        train_loader = data

        for i in range(len(train_loader.dataset.data)):
            label = train_loader.dataset.labels[i] if args.dataset == "svhn" else train_loader.dataset.targets[i]

            if label == class_to_flip:
                if args.dataset == "svhn":
                    train_loader.dataset.labels[i] = target_class
                else:
                    train_loader.dataset.targets[i] = target_class

        return train_loader


def incorrect_labelling(data, args=None):
    if args is None:
        raise ValueError("Args must be provided")

    seed = args.seed
    indices = args.indexes_to_replace

    # Take the data and incorrectly label the forget set, so that in the replacement process it can be replaced with
    # correct labelling

    utils.setup_seed(seed)

    print("Messing with labels")

    os.makedirs(args.backdoor_dir, exist_ok=True)
    dump_file = f"{args.backdoor_dir}/labels.pkl"

    if isinstance(data, OrderedDict):
        # In this case we are in the unlearning step
        # Thus, we want to incorrectly label the forget set and those bits in the train_full set
        # And return a copy of the train_full set in the place of the retain set

        train_loader = data["full"]
        forget_loader = data["forget"]

        retain_loader = copy.deepcopy(train_loader)

        with open(dump_file, "rb+") as f:
            labels = pickle.load(f)

        if args.dataset == "svhn":
            train_loader.dataset.labels[indices] = labels
            forget_loader.dataset.labels = labels
        else:
            train_loader.dataset.targets[indices] = labels
            forget_loader.dataset.targets = labels

        return_dict = OrderedDict(
            full=train_loader, retain=retain_loader, forget=forget_loader, val=data["val"], test=data["test"])

        return return_dict
    else:
        # In this instance data is the train loader and we are in the training phase
        # This function just needs to replace with random labels.

        train_loader = data

        if args.dataset == "svhn" or args.dataset == "cifar10":
            classes = 10
        elif args.dataset == "cifar100":
            classes = 100
        else:
            raise NotImplementedError("This dataset is not implemented")

        class_list = list(set([i for i in range(classes)]))

        for i in indices:
            temp = copy.copy(class_list)
            label = train_loader.dataset.labels[i] if args.dataset == "svhn" else train_loader.dataset.targets[i]
            temp.remove(label)

            new_label = np.random.choice(temp)

            if args.dataset == "svhn":
                train_loader.dataset.labels[i] = new_label
            else:
                train_loader.dataset.targets[i] = new_label

        save_set = train_loader.dataset.labels[indices] if args.dataset == "svhn" else train_loader.dataset.targets[indices]

        # Save labels
        with open(dump_file, 'wb+') as f:
            pickle.dump(save_set, f)

        return train_loader


def grayscale(loader, indices, args):
    dataset = loader.dataset.data

    if args.dataset == "svhn":
        for i in indices:
            channel = np.mean(dataset[i], axis=0)
            dataset[i] = np.stack((channel,) * 3, axis=0)
    else:
        for i in indices:
            channel = np.mean(dataset[i], axis=2)
            dataset[i] = np.stack((channel,) * 3, axis=-1)

    return dataset, dataset[indices]


def feature_replacement(data, args=None):
    if args is None:
        raise ValueError("args must be provided")

    # Take the data (images) and grayscale a percentage of them

    seed = args.seed
    device = args.device

    utils.setup_seed(seed)

    indices = args.indexes_to_replace

    print("Discolouring the data")

    # Given that the grayscale does not depend on the random state of the model and rather the indices, they can be
    # generated each time

    print("Creating the dataset")

    if isinstance(data, OrderedDict):
        train_loader_full = data["full"]

        train_dataset = train_loader_full.dataset

        forget_dataset = copy.deepcopy(train_loader_full.dataset)
        forget_dataset.data = forget_dataset.data[indices]

        retain_images, _ = grayscale(train_loader_full, indices, args)

        retain_dataset = copy.deepcopy(train_dataset)

        for i in indices:
            retain_dataset.data[i] = retain_images[i]

        train_loader_full = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
        )

        retain_loader = torch.utils.data.DataLoader(
            retain_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
        )

        forget_loader = torch.utils.data.DataLoader(
            forget_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
        )

        data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=data["val"], test=data["test"], full=train_loader_full
        )

        return data_loaders
    else:
        train_loader_full = data

        return train_loader_full


def outdated_replacement(data, args=None):
    if args is None:
        raise ValueError("args must be provided")

    # Take the data (images) and grayscale a percentage of them

    seed = args.seed
    device = args.device

    utils.setup_seed(seed)

    indices = args.indexes_to_replace

    print("Discolouring the data")

    # Given that the grayscale does not depend on the random state of the model and rather the indices, they can be
    # generated each time

    print("Creating the dataset")

    if isinstance(data, OrderedDict):
        train_loader_full = data["full"]

        retain_dataset = train_loader_full.dataset

        train_images, forget_images = grayscale(train_loader_full, indices, args)

        for i in indices:
            train_loader_full.dataset.data[i] = train_images[i]

        forget_dataset = copy.deepcopy(train_loader_full.dataset)
        forget_dataset.data = forget_dataset.data[indices]

        retain_loader = torch.utils.data.DataLoader(
            retain_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
        )

        forget_loader = torch.utils.data.DataLoader(
            forget_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
        )

        data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=data["val"], test=data["test"], full=train_loader_full
        )

        return data_loaders
    else:
        train_loader_full = data

        train_images, forget_dataset = grayscale(train_loader_full, indices, args)

        for i in indices:
            train_loader_full.dataset.data[i] = train_images[i]

        # train_loader_full = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
        # )

        return train_loader_full


def noisy_replacement(data, args=None):
    if args is None:
        raise ValueError("args must be provided")

    seed = args.seed
    device = args.device
    noise_level = args.noise_level

    np.random.seed(seed)

    # Collect which indices are being replaced
    indices = args.indexes_to_replace

    print("Noising the dataset")

    save_dir = f"/scratch/ssd004/scratch/newatiaa/noisy_datasets/{args.dataset}_{args.seed}"

    if isinstance(data, OrderedDict):
        print("Loading from file")
        if os.path.exists(save_dir):

            # HAVE TO PACKAGE THE DATASET NOT THE LOADER
            retain_dataset = torch.load(f"{save_dir}/retain.pt")
            forget_dataset = torch.load(f"{save_dir}/forget.pt")
            val_dataset = torch.load(f"{save_dir}/val.pt")
            test_dataset = torch.load(f"{save_dir}/test.pt")
            train_dataset = torch.load(f"{save_dir}/full.pt")

            retain_loader = torch.utils.data.DataLoader(
                retain_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
            )

            forget_loader = torch.utils.data.DataLoader(
                forget_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=False
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
            )

            train_loader_full = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
            )

            data_loaders = OrderedDict(
                retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader, full=train_loader_full
            )

        else:
            os.makedirs(save_dir, exist_ok=True)
            retain_loader = data["retain"]
            forget_loader = data["forget"]
            val_loader = data["val"]
            test_loader = data["test"]
            train_loader_full = data["full"]

            retain_dataset = copy.deepcopy(train_loader_full.dataset)
            retain_loader = torch.utils.data.DataLoader(
                retain_dataset, batch_size=train_loader_full.batch_size, num_workers=0, pin_memory=True, shuffle=True
            )

            # Add noise to the training dataset at the specified indices

            # Add noise to the training dataset at the specified indices
            for i in indices:
                # utils.display_image(train_loader_full.dataset.data[i])
                if args.dataset == "2d_synthetic":
                    if args.noise_type == "gaussian":
                        modifier = np.random.normal(0,
                                                    torch.abs_(
                                                        noise_level * torch.max(train_loader_full.dataset.data[i])),
                                                    train_loader_full.dataset.data[i].shape)
                    elif args.noise_type == "salt_and_pepper":
                        modifier = np.random.choice([0, 255], size=train_loader_full.dataset.data[i].shape,
                                                    p=[noise_level, 1 - noise_level])
                    elif args.noise_type == "poisson":
                        modifier = np.random.poisson(noise_level * torch.max(train_loader_full.dataset.data[i]),
                                                     train_loader_full.dataset.data[i].shape)
                    else:
                        raise ValueError("Invalid replacement type")
                else:
                    if args.noise_type == "gaussian":
                        modifier = np.random.normal(0, noise_level * np.max(train_loader_full.dataset.data[i]),
                                                    train_loader_full.dataset.data[i].shape)
                    elif args.noise_type == "salt_and_pepper":
                        modifier = np.random.choice([0, 255], size=train_loader_full.dataset.data[i].shape,
                                                    p=[noise_level, 1 - noise_level])
                    elif args.noise_type == "poisson":
                        modifier = np.random.poisson(noise_level * np.max(train_loader_full.dataset.data[i]),
                                                     train_loader_full.dataset.data[i].shape)
                    else:
                        raise ValueError("Invalid replacement type")

                train_loader_full.dataset.data[i] = train_loader_full.dataset.data[i] + modifier

            assert (
                    retain_loader.dataset != train_loader_full.dataset)  # Ensure that the retain dataset is
            # different from the full dataset

            # Create the forget loader
            forget_dataset = copy.deepcopy(train_loader_full.dataset)
            forget_dataset.data = forget_dataset.data[indices]
            forget_loader = torch.utils.data.DataLoader(
                forget_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True
            )

            data_loaders = OrderedDict(
                retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader, full=train_loader_full
            )

            torch.save(retain_loader.dataset, f"{save_dir}/retain.pt")
            torch.save(forget_loader.dataset, f"{save_dir}/forget.pt")
            torch.save(val_loader.dataset, f"{save_dir}/val.pt")
            torch.save(test_loader.dataset, f"{save_dir}/test.pt")
            torch.save(train_loader_full.dataset, f"{save_dir}/full.pt")

        # Return the ordered dict with the new retain dataset
        return data_loaders
    else:
        print("Creating the dataset")
        train_loader_full = data

        # Add noise to the training dataset at the specified indices
        for i in indices:
            # utils.display_image(train_loader_full.dataset.data[i])
            if args.dataset == "2d_synthetic":
                if args.noise_type == "gaussian":
                    modifier = np.random.normal(0,
                                                torch.abs_(noise_level * torch.max(train_loader_full.dataset.data[i])),
                                                train_loader_full.dataset.data[i].shape)
                elif args.noise_type == "salt_and_pepper":
                    modifier = np.random.choice([0, 255], size=train_loader_full.dataset.data[i].shape,
                                                p=[noise_level, 1 - noise_level])
                elif args.noise_type == "poisson":
                    modifier = np.random.poisson(noise_level * torch.max(train_loader_full.dataset.data[i]),
                                                 train_loader_full.dataset.data[i].shape)
                else:
                    raise ValueError("Invalid replacement type")
            else:
                if args.noise_type == "gaussian":
                    modifier = np.random.normal(0, noise_level * np.max(train_loader_full.dataset.data[i]),
                                                train_loader_full.dataset.data[i].shape)
                elif args.noise_type == "salt_and_pepper":
                    modifier = np.random.choice([0, 255], size=train_loader_full.dataset.data[i].shape,
                                                p=[noise_level, 1 - noise_level])
                elif args.noise_type == "poisson":
                    modifier = np.random.poisson(noise_level * np.max(train_loader_full.dataset.data[i]),
                                                 train_loader_full.dataset.data[i].shape)
                else:
                    raise ValueError("Invalid replacement type")

            train_loader_full.dataset.data[i] = train_loader_full.dataset.data[i] + modifier
            # utils.display_image(train_loader_full.dataset.data[i])

        return train_loader_full
