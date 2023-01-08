import torchvision
from torchvision import datasets, transforms
from src.model import MLModel
from src.trainer import Trainer
from src.dataloader import Loader

training_dir='cifar10-dataset'
# Define data augmentation
def _get_transforms():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return

train_set = torchvision.datasets.CIFAR10(root=training_dir,
                                         train=True,
                                         download=False,
                                         transform=_get_transforms())

val_set = torchvision.datasets.CIFAR10(root=training_dir,
                                        train=False,
                                        download=False,
                                        transform=_get_transforms())

datasets = (train_set, val_set)
model = MLModel()
config = {
    'epochs':15,
    'seed': 32,
    'batch_size': 32,
    'scheduler': None,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'lr': 0.001,
    'criterion': 'cross_entropy',
    'metric': 'accuracy',
    'model_dir': 'model_output'
}
trainer = Trainer(model, datasets, config, is_parallel=False)
trainer.fit()
