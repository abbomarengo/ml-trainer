import torch
from torch.utils.data import distributed
from torch.nn import parallel
from torch import optim
from sklearn.metrics import accuracy_score
import structlog
import os
import gc
from tqdm import tqdm

# SageMaker data parallel: Import the library PyTorch API
# import smdistributed.dataparallel.torch.torch_smddp

# SageMaker data parallel: Import PyTorch's distributed API
import torch.distributed as dist

# Local import
from .dataloader import Loader

logger = structlog.get_logger('__name__')


class Trainer():
    def __init__(self, model, datasets, config, is_parallel=True, backend='smddp'):
        train_set, val_set = datasets
        torch.manual_seed(config['seed'])
        self.model = model
        self.config = config
        self.is_parallel = is_parallel
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        logger.info("Loading the model.")
        if self.is_parallel:
            dist.init_process_group(backend=backend)
            train_sampler = distributed.DistributedSampler(self.train_loader, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())
            # Scale batch size by world size
            batch_size = config['batch_size'] // dist.get_world_size()
            batch_size = max(batch_size, 1)
            assert torch.device("cuda" if torch.cuda.is_available() else "cpu") == "cuda", \
                "Need GPU availability for data parallelism."
            self.device = torch.device("cuda")
        else:
            train_sampler = None
            batch_size = config['batch_size']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        if self.config['scheduler'] != None:
            self.scheduler = self.scheduler_options[self.config['scheduler']]
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

    def _get_optimizer(self):
        if self.config['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])

    def _get_criterion(self):
        if self.config['criterion'] == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()

    def _average_gradients(self):
        # Average gradients (only for multi-node CPU)
        # Gradient averaging.
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def _evaluate(self, outputs, targets):
        if self.config['metric'] == 'mcrmse':
            colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
            return torch.mean(torch.sqrt(colwise_mse), dim=0)
        if self.config['metric'] == 'accuracy':
            return accuracy_score(targets, outputs)

    def _get_predictions(self, outputs):
        return outputs

    def _train_one_epoch(self, epoch):
        # Need to change the tqdm
        self.model.train()
        running_loss = 0.
        progress = tqdm(self.train_loader, total=len(self.train_loader))
        for i, (inputs, targets) in enumerate(progress):
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
                self.scheduler.step(epoch - 1 + i / len(self.train_loader))  # as per pytorch docs
            del inputs, targets, outputs, loss
        if self.config['scheduler'] == 'StepLR':
            self.scheduler.step()
        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)

    @torch.no_grad()
    def _validate_one_epoch(self):
        running_loss = 0.
        running_metric = 0.
        progress = tqdm(self.val_loader, total=len(self.val_loader))
        for (inputs, targets) in progress:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item()
            if self.config['metric'] != None:
                outputs = self._get_predictions(outputs)
                running_metric += self._evaluate(outputs, targets).item()
            del inputs, targets, outputs, loss
        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_metrics.append(running_metric / len(self.val_loader))
        del running_metric
        if self.config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler.step(val_loss)

    def _save_model(self, model_dir):
        logger.info("Saving the model.")
        path = os.path.join(model_dir, "model.pth")
        torch.save(self.model.cpu().state_dict(), path)

    def fit(self):
        logger.info("Start training..")
        fit_progress = tqdm(range(1, self.config['epochs'] + 1), leave=True, desc="Training...")
        for epoch in fit_progress:
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self._train_one_epoch(epoch)
            self.clear()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self._validate_one_epoch()
            self.clear()
            # Save model on master node.
            if self.is_parallel:
                if dist.get_rank() == 0:
                    self._save_model(self.config['model_dir'])
            else:
                self._save_model(self.config['model_dir'])
            logger.info(f"{'-' * 30} EPOCH {epoch} / {self.config['epochs']} {'-' * 30}")
            logger.info(f"train loss: {self.train_losses[-1]}")
            logger.info(f"valid loss: {self.val_losses[-1]} - valid metric: {self.val_metrics[-1]}\n\n")
        logger.info("Training Complete.")

    def test(self, test_loader):
        preds = []
        for (inputs) in test_loader:
            inputs = {k: inputs[k].to(self.device) for k in inputs.keys()}
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())
        preds = torch.concat(preds)
        return preds

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()
