import torch
from torch.utils.data import distributed
from torch.nn import parallel, LogSoftmax, Softmax
from torch import optim
from sklearn.metrics import accuracy_score
import structlog
import os
import gc
from tqdm import tqdm
import pickle

# SageMaker data parallel: Import PyTorch's distributed API
import torch.distributed as dist

# Local import
from .dataloader import Loader
from .utils.functions import custom_loss_function

logger = structlog.get_logger('__name__')


class Trainer():
    def __init__(self, model, datasets=None, epochs=None, batch_size=None,
                 is_parallel=False, save_history=False, **config):
        logger.info('Config inputs.', config=config)
        allowed_kwargs = {"seed", "scheduler", "optimizer", "momentum", "weight_decay",
                          "lr", "criterion", "metric", "pred_function", "model_dir", "backend"}
        self.validate_kwargs(config, allowed_kwargs)
        # Unpack kwargs
        self.epochs = epochs
        self.scheduler_type = config.get('scheduler', None)
        self.optimizer_type = config.get('optimizer', 'sgd')
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.lr = config.get('lr', 0.001)
        self.criterion_type = config.get('criterion', 'cross_entropy')
        self.metric = config.get('metric', 'accuracy')
        self.pred_function_type = config.get('pred_function', 'softmax')
        self.model_dir = config.get('model_dir', 'model_output')
        backend = config.get('backend', 'smddp')
        seed = config.get('seed', 32)
        # SageMaker data parallel: Import the library PyTorch API
        if is_parallel:
            import smdistributed.dataparallel.torch.torch_smddp
        if datasets:
            train_set, val_set = datasets
        torch.manual_seed(seed)
        self.model = model
        self.is_parallel = is_parallel
        self.save_history = save_history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.history = {}
        logger.info("Loading the model.")
        if self.is_parallel:
            if datasets:
                dist.init_process_group(backend=backend)
                train_sampler = distributed.DistributedSampler(train_set, num_replicas=dist.get_world_size(),
                                                               rank=dist.get_rank())
                # Scale batch size by world size
                batch_size = batch_size // dist.get_world_size()
                batch_size = max(batch_size, 1)
            else:
                logger.warning("Testing only available. No datasets in arguments.")
        else:
            if datasets:
                train_sampler = None
            else:
                logger.warning("Testing only available. No datasets in arguments.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Training on device: {self.device}.')
        if datasets:
            logger.info('Loading training and validation set.')
            logger.info("Preparing the data.")
            self.train_loader = Loader(train_set, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler)
            self.val_loader = Loader(val_set, batch_size=batch_size, shuffle=True)
            logger.debug(
                "Processes {}/{} ({:.0f}%) of train data".format(
                    len(self.train_loader.sampler),
                    len(self.train_loader.dataset),
                    100.0 * len(self.train_loader.sampler) / len(self.train_loader.dataset),
                )
            )
            logger.debug(
                "Processes {}/{} ({:.0f}%) of validation data".format(
                    len(self.val_loader.sampler),
                    len(self.val_loader.dataset),
                    100.0 * len(self.val_loader.sampler) / len(self.val_loader.dataset),
                )
            )
        else:
            logger.warning("Testing only available. No datasets in arguments.")
        self.model = self.model.to(self.device)
        if self.is_parallel:
            self.model = parallel.DistributedDataParallel(self.model)
            local_rank = os.environ["LOCAL_RANK"]
            torch.cuda.set_device(int(local_rank))
            self.model.cuda(int(local_rank))
        criterion = self._get_criterion()
        self.criterion = criterion.to(self.device)
        self.optimizer = self._get_optimizer()
        self.scheduler_options = {
            'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5,
                                                                                                eta_min=1e-7),
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-7),
            'StepLR': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2)
        }
        if self.scheduler_type:
            self.scheduler = self.scheduler_options[self.scheduler_type]
        self.pred_function = self._get_prediction_function()

    def _get_prediction_function(self):
        if self.pred_function_type == 'logsoftmax':
            return LogSoftmax(dim=-1)
        elif self.pred_function_type == 'softmax':
            return Softmax(dim=-1)
        else:
            return None

    def _get_optimizer(self):
        if self.optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr,
                             momentum=self.momentum, weight_decay=self.weight_decay)
        if self.optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr,
                              weight_decay=self.weight_decay)
        if self.optimizer_type == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
        if self.optimizer_type == 'adamax':
            return optim.Adamax(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        if self.optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

    def _get_criterion(self):
        if self.criterion_type == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        if self.criterion_type == 'neg-loss':
            return torch.nn.NLLLoss
        if self.criterion_type == 'l1':
            return torch.nn.L1Loss()
        if self.criterion_type == 'l2':
            return torch.nn.MSELoss
        if self.criterion_type == 'custom':
            return custom_loss_function

    def _average_gradients(self):
        # Average gradients (only for multi-node CPU)
        # Gradient averaging.
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def _evaluate(self, outputs, targets):
        if self.metric == 'mcrmse':
            colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
            return torch.mean(torch.sqrt(colwise_mse), dim=0)
        if self.metric == 'accuracy':
            predictions = self._get_predictions(outputs)
            return accuracy_score(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())

    def _get_predictions(self, outputs):
        if self.pred_function_type:
            return torch.argmax(self.pred_function(outputs), dim=-1)
        else:
            return torch.argmax(outputs, dim=-1)

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.model = self.model.to(self.device)
        running_loss = 0.
        running_metric = 0.
        with tqdm(self.train_loader, unit='batch') as tepoch:
            for i, (inputs, targets) in enumerate(tepoch):
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if self.scheduler_type == 'CosineAnnealingWarmRestarts':
                    self.scheduler.step(epoch - 1 + i / len(self.train_loader))  # as per pytorch docs
                if self.metric:
                    running_metric += self._evaluate(outputs, targets)
                    tepoch.set_postfix(loss=running_loss / len(self.train_loader),
                                       metric=running_metric / len(self.train_loader))
                else:
                    tepoch.set_postfix(loss=loss.item())
                del inputs, targets, outputs, loss
        if self.scheduler_type == 'StepLR':
            self.scheduler.step()
        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)
        if self.metric:
            self.train_metrics.append(running_metric / len(self.train_loader))
        del running_metric

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        running_loss = 0.
        running_metric = 0.
        with tqdm(self.val_loader, unit='batch') as tepoch:
            for (inputs, targets) in tepoch:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                if self.metric:
                    running_metric += self._evaluate(outputs, targets)
                    tepoch.set_postfix(loss=running_loss / len(self.val_loader),
                                       metric=running_metric / len(self.val_loader))
                else:
                    tepoch.set_postfix(loss=loss.item())
                del inputs, targets, outputs, loss
        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
        if self.metric:
            self.val_metrics.append(running_metric / len(self.val_loader))
        del running_metric

    def save_model(self, model_dir):
        logger.info("Saving the model.")
        path = os.path.join(model_dir, "model.pth")
        torch.save(self.model.cpu().state_dict(), path)

    def save_history_(self, model_dir):
        logger.info("Saving the training history.")
        path = os.path.join(model_dir, "history.pkl")
        with open(path, "wb") as fp:
            pickle.dump(self.history, fp)

    def fit(self):
        logger.info("Start training..")
        for epoch in range(1, self.epochs + 1):
            logger.info(f"{'-' * 30} EPOCH {epoch} / {self.epochs} {'-' * 30}")
            self._train_one_epoch(epoch)
            self.clear()
            self._validate_one_epoch()
            self.clear()
            # Save model on master node.
            if self.is_parallel:
                if dist.get_rank() == 0:
                    self.save_model(self.model_dir)
            else:
                self.save_model(self.model_dir)
            if self.metric:
                logger.info(f"train loss: {self.train_losses[-1]} - "
                            f"train {self.metric}: {self.train_metrics[-1]}")
                logger.info(f"valid loss: {self.val_losses[-1]} - "
                            f"valid {self.metric}: {self.val_metrics[-1]}\n\n")
            else:
                logger.info(f"train loss: {self.train_losses[-1]}")
                logger.info(f"valid loss: {self.val_losses[-1]}\n\n")
        self.history = {
            'epochs': [*range(1, self.epochs + 1)],
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_metric': self.train_metrics,
            'val_metric': self.val_metrics,
            'metric_type': self.metric
        }
        if self.save_history:
            self.save_history_(self.model_dir)
        logger.info("Training Complete.")

    def test(self, model, test_loader):
        logger.info("Testing..")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        running_loss = 0.
        running_metric = 0.
        with tqdm(test_loader, unit='batch') as tepoch:
            for (inputs, targets) in tepoch:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                if self.metric:
                    running_metric += self._evaluate(outputs, targets)
                    tepoch.set_postfix(loss=running_loss / len(test_loader),
                                       metric=running_metric / len(test_loader))
                else:
                    tepoch.set_postfix(loss=loss.item())
                del inputs, targets, outputs, loss
        test_loss = running_loss / len(test_loader)
        if self.metric:
            test_metric = running_metric / len(test_loader)
            return test_loss, test_metric
        return test_loss

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()

    def validate_kwargs(self, kwargs, allowed_kwargs, error_message="Keyword argument not understood:"):
        """Checks that all keyword arguments are in the set of allowed keys."""
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError(error_message, kwarg)
