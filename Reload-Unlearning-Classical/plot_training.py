import torch
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    logs = {}
    logd_path = "./logs"

    # Load all the logs from the folder
    for log in os.listdir(logd_path):
        # If log is a directory, skip it
        if os.path.isdir(logd_path + "/" + log):
            continue
        logs[log] = torch.load(logd_path + "/" + log)

        # Remove the .pt extension
        dir_name = log.split(".")[0]

        # Make a folder with the same name as the log file
        os.makedirs(f"./logs/{dir_name}", exist_ok=True)

        if "RELOAD" not in logs[log]:
            print(f"RELOAD not found in {log}")
            continue

        zero_logs = logs[log]["RELOAD"]

        forget_losses = zero_logs["forget"]["loss"].vals
        forget_accs = zero_logs["forget"]["accuracy"].vals

        retain_losses = zero_logs["retain"]["loss"].vals
        retain_accs = zero_logs["retain"]["accuracy"].vals

        epochs = zero_logs["epochs"]

        # Plot the loss values
        plt.figure()
        plt.plot(range(epochs), forget_losses, label="Forget Loss")
        plt.plot(range(epochs), retain_losses, label="Retain Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("ZERO Loss Value Progression")
        plt.legend()
        plt.savefig(f"./logs/{dir_name}/zero_loss.png", dpi=300)

        plt.close()

        # Plot the accuracy values
        plt.figure()
        plt.plot(range(epochs), forget_accs, label="Forget Accuracy")
        plt.plot(range(epochs), retain_accs, label="Retain Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("ZERO Accuracy Progression")
        plt.legend()
        plt.savefig(f"./logs/{dir_name}/zero_accuracy.png", dpi=300)

        plt.close()

        if "retrain" not in logs[log]:
            print(f"retrain not found in {log}")
            continue

        retrain_logs = logs[log]["retrain"]

        forget_losses = retrain_logs["forget"]["loss"].vals
        forget_accs = retrain_logs["forget"]["accuracy"].vals

        retain_losses = retrain_logs["retain"]["loss"].vals
        retain_accs = retrain_logs["retain"]["accuracy"].vals

        epochs = [i for i in range(len(forget_losses))]

        # Plot the loss values
        plt.figure()
        plt.plot(epochs, forget_losses, label="Forget Loss")
        plt.plot(epochs, retain_losses, label="Retain Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Retrain Loss Value Progression")
        plt.legend()
        plt.savefig(f"./logs/{dir_name}/retrain_loss.png", dpi=300)

        plt.close()

        # Plot the accuracy values
        plt.figure()
        plt.plot(epochs, forget_accs, label="Forget Accuracy")
        plt.plot(epochs, retain_accs, label="Retain Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Retrain Accuracy Progression")
        plt.legend()
        plt.savefig(f"./logs/{dir_name}/retrain_accuracy.png", dpi=300)

        plt.close()

        print(f"Plots saved for {log}")