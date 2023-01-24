import pickle
import os
from matplotlib import pyplot as plt


def load_history(model_dir):
    path = os.path.join(model_dir, 'history', "history.pkl")
    history = pickle.load(open(path, "rb"))
    return history


def plot_history(history: dict) -> None:
    x = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    if history['metric_type'] != None:
        train_metric = history['train_metric']
        val_metric = history['val_metric']
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(x, train_loss, c='C0', label='train')
        axes[0].plot(x, val_loss, c='C1', label='validation')
        axes[1].plot(x, train_metric, c='C0', label='train')
        axes[1].plot(x, val_metric, c='C1', label='validation')
        axes[0].set_xticks(x)
        axes[1].set_xticks(x)
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel(history['metric_type'])
        axes[0].set_title('Training Loss vs. Validation Loss')
        axes[1].set_title(f"{history['metric_type']} - Training vs. Validation")
    else:
        plt.subplots(figsize=(10, 5))
        plt.plot(x, train_loss, c='C0', label='train')
        plt.plot(x, val_loss, c='C1', label='validation')
        plt.xticks(x)
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
