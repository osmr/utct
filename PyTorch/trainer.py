import os
import logging

from utct.common.trainer_template import TrainerTemplate

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Trainer(TrainerTemplate):
    """
    Class, which provides training process under PyTorch framework.

    Parameters:
    ----------
    model : object
        instance of Model class with graph of CNN
    optimizer : object
        instance of Optimizer class with CNN optimizer
    data_source : object
        instance of DataSource class with training/validation iterators
    saver : object
        instance of Saver class with information about stored files
    cuda : bool
        use CUDA
    """
    def __init__(self,
                 model,
                 optimizer,
                 data_source,
                 saver,
                 cuda=False):
        super(Trainer, self).__init__(
            model,
            optimizer,
            data_source,
            saver)
        self.cuda = cuda and torch.cuda.is_available()

    def _hyper_train_target_sub(self, **kwargs):
        """
        Calling single training procedure for specific hyper parameters from hyper optimizer.
        """

        if self.saver.log_filename:
            fh = logging.FileHandler(self.saver.log_filename)
            self.logger.addHandler(fh)

        self.logger.info("Training with parameters: {}".format(kwargs))

        train_loader, val_loader = self.data_source()

        model = self.model()
        if self.cuda:
            model.cuda()

        optimizer = self.optimizer(
            params=model.parameters(),
            **kwargs)

        for epoch in range(1, self.num_epoch + 1):
            self._train_epoch(
                epoch=epoch,
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                cuda=self.cuda,
                logger=self.logger)
            self._val_epoch(
                model=model,
                data_loader=val_loader,
                cuda=self.cuda,
                logger=self.logger)

        if self.saver.log_filename:
            self.logger.removeHandler(fh)
            fh.close()

        best_value = 0.0

        return best_value

    @staticmethod
    def _train_epoch(epoch,
                     model,
                     data_loader,
                     optimizer,
                     log_interval=1,
                     cuda=False,
                     logger=None):
        model.train()
        pid = os.getpid()
        for batch_idx, (data, target) in enumerate(data_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if (batch_idx % log_interval == 0) and (logger is not None):
                logger.info('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid,
                    epoch,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.data[0]))

    @staticmethod
    def _val_epoch(model,
                   data_loader,
                   cuda=False,
                   logger=None):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            #pred = output.data.max(1)[1] # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #correct += pred.eq(target.data).cpu().sum()
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(data_loader.dataset)
        if logger is not None:
            logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss,
                correct,
                len(data_loader.dataset),
                100.0 * correct / len(data_loader.dataset)))

