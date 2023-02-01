import pickle
import os
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import torch


def load_history(file_dir):
    path = os.path.join(file_dir, "history.pkl")
    history = pickle.load(open(path, "rb"))
    return history


def load_model(model, PATH):
    # If model was trained in parallel
    try:
        state_dict = torch.load(PATH)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    except Exception as e:
        model.load_state_dict(torch.load(PATH))
    return model


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
        if len(x) > 25:
            axes[0].set_xticks(np.arange(0, len(x)+1, 5))
            axes[0].set_xticklabels(np.arange(0, len(x)+1, 5), rotation=45)
            axes[1].set_xticks(np.arange(0, len(x)+1, 5))
            axes[1].set_xticklabels(np.arange(0, len(x)+1, 5), rotation=45)
        else:
            axes[0].set_xticks(x)
            axes[1].set_xticks(x)
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel(history['metric_type'])
        axes[0].set_title('Training Loss vs. Validation Loss')
        axes[1].set_title(f"{history['metric_type']} - Training vs. Validation")
        axes[0].legend()
        axes[1].legend()
    else:
        plt.subplots(figsize=(10, 5))
        plt.plot(x, train_loss, c='C0', label='train')
        plt.plot(x, val_loss, c='C1', label='validation')
        plt.xticks(x, rotation=45)
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Validation Loss')
        plt.legend()
    plt.tight_layout()
    plt.show()
