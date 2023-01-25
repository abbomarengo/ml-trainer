import argparse
import os
import json
import torchvision
from src.model import MLModel
from src.trainer import Trainer


def main(args):
    # Insert CODE HERE:
    # EXAMPLE
    if args.custom_function:
        from src.utils.functions import custom_pre_process_function
        train_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                                 train=True,
                                                 download=False,
                                                 transform=custom_pre_process_function())
        val_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                               train=False,
                                               download=False,
                                               transform=custom_pre_process_function())
    else:
        train_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                                 train=True,
                                                 download=False)
        val_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                               train=False,
                                               download=False)
    datasets = (train_set, val_set)
    model = MLModel()
    config = {
        'epochs': args.epochs,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'scheduler': args.scheduler,
        'optimizer': args.optimizer,
        'momentum': args.momentum,
        'lr': args.lr,
        'criterion': args.criterion,
        'metric': args.metric,
        'model_dir': args.model_dir
    }
    trainer = Trainer(model, datasets, config, is_parallel=True, save_history=True, backend=args.backend)
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Pytorch environments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="input batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--optimizer", type=str, default='sgd',
                        help="optimizer for backward pass (default: sgd)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.010)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (default: 0.9)")
    parser.add_argument("--seed", type=int, default=32,
                        help="random seed (default: 32)")
    parser.add_argument("--scheduler", type=str, default=None,
                        help="apply scheduler for learning rate (default: None)")
    parser.add_argument("--criterion", type=str, default='cross_entropy',
                        help="loss function to apply (default: cross_entropy)")
    parser.add_argument("--metric", type=str, default=None,
                        help="metric for model evaluation (default: None)")
    parser.add_argument("--backend", type=str, default='smddp',
                        help="backend for dist. training, this script only supports gloo")
    parser.add_argument("--custom_function", type=bool, default=False,
                        help="apply a pre-processing function (default: False)")

    # SageMaker environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    main(parser.parse_args())
