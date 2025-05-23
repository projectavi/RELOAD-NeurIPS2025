# Import required libraries
import os
import fcntl

import json
import sys

import time
import subprocess

if __name__ == "__main__":
    # List out all of the experiments

    architectures = ["resnet18", "vgg16_bn"]
    datasets = ["cifar10", "cifar100", "svhn"]

    tasks_tup = [
        ("replace", "poisoning", "distributed_pattern"),
        ("replace", "poisoning", "feature_pattern"),
        ("replace", "poisoning", "cross_pattern"),
        ("replace", "targeted_label_flip", ""),
        ("replace", "incorrect_label", ""),
        ("replace", "noisy", "gaussian"),
        ("remove", "0.1", ""),
        ("remove", "0.2", ""),
        ("remove", "0.3", ""),
        ("remove_inclass", "0.1", ""),
        ("remove_inclass", "0.2", ""),
        ("knn_forget", "0.05", ""),
    ]

    tasks = [f"{tup[0]};{tup[1]};{tup[2]}" for tup in tasks_tup]

    if os.path.exists("./experiments.json"):
        f = open('./experiments.json', 'r+', encoding='utf-8')
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                break
            except:
                time.sleep(0.1)
        experiments = json.load(f)
    else:
        f = open('./experiments.json', 'w+', encoding='utf-8')
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                break
            except:
                time.sleep(0.1)
        experiments = {}

        for arch in architectures:
            experiments[arch] = {}
            for dataset in datasets:
                experiments[arch][dataset] = {}
                for task in tasks:
                    task_tup = task.split(";")
                    experiments[arch][dataset][task] = {"STATE": "TBD", "ID": ""}
                    if arch == "resnet18" and (task_tup[1] in ["incorrect_label", "targeted_label_flip", "noisy"] or (task_tup[1] == "poisoning" and task_tup[2] in ["cross_pattern", "distributed_pattern"])):
                        experiments[arch][dataset][task] = {"STATE": "COMPLETE", "ID": ""}
        # Dump experiments to file

        json.dump(experiments, f, ensure_ascii=False, indent=4)
        exit()

    # Select an experiment to do and dump this to the log
    task_ID = None
    found = False
    for arch in architectures:
        if found:
            break
        for dataset in experiments[arch].keys():
            if found:
                break
            for task in experiments[arch][dataset].keys():
                if task in ["knn_forget"]:
                    pass # JUST FOR NOW RUN EVERYTHING ELSE
                if experiments[arch][dataset][task]["STATE"] == "TBD":
                    # Choose this task

                    task_tup = task.split(";")

                    chosen_task = (arch, dataset, task_tup[0], task_tup[1], task_tup[2])
                    task_ID = experiments[arch][dataset][task]["ID"]
                    experiments[arch][dataset][task]["STATE"] = "IPR"
                    found = True
                    break
    if task_ID is None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        exit()

    f.seek(0)
    json.dump(experiments, f, ensure_ascii=False, indent=4)
    f.truncate()
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    f.close()
    trained_model_dir = "/scratch/ssd004/scratch/newatiaa/models"
    log_dir = "/scratch/ssd004/scratch/newatiaa/logs"

    arch = chosen_task[0]
    dataset = chosen_task[1]
    task = chosen_task[2]
    detail1 = chosen_task[3]
    detail2 = chosen_task[4]

    name = ""
    for item in chosen_task:
        if len(item) > 0:
            if item[0] == "0":
                name += item[2].upper()
            else:
                name += item[0].upper()
        else:
            name += ""

    submitted = False
    print(f"SELECTED TASK: {chosen_task} with ID: {task_ID} and name {name}")
    sys.stdout.flush()
    # A task is selected. The first thing to do is train the models. For this we just submit batch jobs to m2 and m3 qos. There will be 10 training runs. Submit 4 to m2 and 6 to m3.
    if task == "replace":
        # First check how many of the trained models exist
        # These will be under /scratch/ssd004/scratch/newatiaa/models
        seeds = [i for i in range(0, 10)]

        seeds_trained = []
        for seed in seeds:
            trained_model_filename = f"0{arch}_{seed}_{dataset}_{detail1}_0"
            if detail1 == "noisy":
                trained_model_filename += f"_{detail2}_0.4.pth.tar"
            elif detail1 == "poisoning":
                trained_model_filename += f"_{detail2}.pth.tar"
            else:
                    trained_model_filename+= ".pth.tar"

            if os.path.exists(f"{trained_model_dir}/{trained_model_filename}"):
                print(f"{trained_model_dir}/{trained_model_filename}")
                seeds_trained.append(seed)
            else:
                print(f"TRAINING FOR SEED {seed}")

            sys.stdout.flush()
                    
        while len(seeds_trained) != len(seeds):
            if submitted:
                seeds_trained = []
                for seed in seeds:
                    trained_model_filename = f"0{arch}_{seed}_{dataset}_{detail1}_0"
                    if detail1 == "noisy":
                        trained_model_filename += f"_{detail2}_0.4.pth.tar"
                    elif detail1 == "poisoning":
                        trained_model_filename += f"_{detail2}.pth.tar"
                    else:
                        trained_model_filename+= ".pth.tar"

                    if os.path.exists(f"{trained_model_dir}/{trained_model_filename}"):
                        seeds_trained.append(seed)

                    # OPTIONAL move this to checkpoint and override in actual script to read
            else:
                # Submit 10 jobs for training
                for i in range(0, 4):
                    subprocess.Popen(["sbatch", "-J", f"{name}_T", "m2_format.sh", task_ID])
                for i in range(0, 6):
                    subprocess.Popen(["sbatch", "-J", f"{name}_T", "m3_format.sh", task_ID])

                submitted = True

                print(f"TRAINING STARTED")

        print(f"ALL TRAINED MODELS FOUND")

        for seed in seeds:
            trained_model_filename = f"0{arch}_{seed}_{dataset}_{detail1}_0"
            if detail1 == "noisy":
                trained_model_filename += f"_{detail2}_0.4"
            elif detail1 == "poisoning":
                trained_model_filename += f"_{detail2}"

            os.makedirs("./logs", exist_ok=True)
            if os.path.exists(f"./logs/{trained_model_filename}_MARKER.pt"):
                # Delete the file
                os.remove(f"./logs/{trained_model_filename}_MARKER.pt")

        print(f"MARKERS REMOVED")

        # ONCE THIS IS EXITED TRAINING IS COMPLETE AND NOW UNLEARNING CAN BEGIN
        for i in range(0, 20):
            subprocess.Popen(["sbatch", "-J", f"{name}_R", "m4_format.sh", task_ID])
            if i % 4 == 0:
                subprocess.Popen(["sbatch", "-J", f"{name}_R", "deadline_format.sh", task_ID])
            time.sleep(1)


        print(f"UNLEARNING JOBS SUBMITTED")
    else:
        # Train once - so first check if this exists. This will be in the home directory trained models.
        trained_model_path = f"./trained_models/0{dataset}_{arch}_model_SA_best.pth.tar"

        if os.path.exists(trained_model_path):
            print(f"TRAINED MODEL FOUND")
            # CHECK IF THE RETRAINED MODELS EXIST FOR EACH SEED AND IF NOT THEN BATCH SUBMIT THEM, 4 on m2, 6 on m3
            retrained_model_dir = f"/scratch/ssd004/scratch/newatiaa/logs"
            seeds = [i for i in range(1, 11)]

            seeds_trained = []
            for seed in seeds:
                trained_model_filename = f"{arch}_{detail1}_{task}_{dataset}_{seed}_retrain.pt"

                if os.path.exists(
                        f"{retrained_model_dir}/{trained_model_filename}"):
                    seeds_trained.append(seed)
                else:
                    print(f"Retraining for {seed}, {retrained_model_dir}/{trained_model_filename} not found")
                    sys.stdout.flush()

                        
            while len(seeds_trained) != len(seeds):
                if submitted:
                    seeds_trained = []
                    for seed in seeds:
                        trained_model_filename =  f"{arch}_{detail1}_{task}_{dataset}_{seed}_retrain.pt"

                        if os.path.exists(
                                f"{retrained_model_dir}/{trained_model_filename}"):
                            seeds_trained.append(seed)

                        # OPTIONAL move this to checkpoint and override in actual script to read
                else:
                    # Submit 10 jobs for training
                    for i in range(0, 4):
                        subprocess.Popen(["sbatch", "-J", f"{name}_T", "m2_format.sh", task_ID])
                    for i in range(0, 6):
                        subprocess.Popen(["sbatch", "-J", f"{name}_T", "m3_format.sh", task_ID])

                    print("RETRAINING JOBS SUBMITTED")

                    submitted = True

            print(f"RETRAINED MODELS FOUND")

            # UNLEARNING BATCH SUBMIT
            for i in range(0, 20):
                subprocess.Popen(["sbatch", "-J", f"{name}_U", "m4_format.sh", task_ID])

            print(f"UNLEARNING JOBS SUBMITTED")

    f = open('./experiments.json', 'r+', encoding='utf-8')
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            break
        except:
            time.sleep(0.1)
    experiments = json.load(f)
    found = False
    print(task_ID)
    # Select an experiment to do and dump this to the log
    for arch in experiments.keys():
        if found:
            break
        for dataset in experiments[arch].keys():
            if found:
                break
            for task in experiments[arch][dataset].keys():
                if experiments[arch][dataset][task]["ID"] == task_ID:
                    # Choose this task
                    print("FOUND")

                    experiments[arch][dataset][task]["STATE"] = "UNLEARNING"
                    found = True
                    break
    print(experiments)
    f.seek(0)
    json.dump(experiments, f, ensure_ascii=False, indent=4)
    f.truncate()
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    f.close()

    print("JSON FILE UPDATED, EXITING")

    exit()