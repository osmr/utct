import logging

from utct.common.trainer_template import TrainerTemplate

import mxnet as mx
from mxnet import gluon, autograd


class Trainer(TrainerTemplate):
    """
    Class, which provides training process under Gluon/MXNet framework.

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
    ctx : object
        instance of MXNet context
    """
    def __init__(self,
                 model,
                 optimizer,
                 data_source,
                 saver,
                 ctx):
        super(Trainer, self).__init__(
            model,
            optimizer,
            data_source,
            saver)
        self.ctx = ctx

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
        model.initialize(
            mx.init.Xavier(magnitude=2.24),
            ctx=self.ctx)

        trainer = self.optimizer(
            params=model.collect_params(),
            **kwargs)

        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        log_interval = 1
        for epoch in range(self.num_epoch):
            metric.reset()
            for i, (data, label) in enumerate(train_loader):
                # Copy data to ctx if necessary
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                # Start recording computation graph with record() section.
                # Recorded graphs can then be differentiated with backward.
                with autograd.record():
                    output = model(data)
                    L = loss(output, label)
                    L.backward()
                # take a gradient step with batch_size equal to data.shape[0]
                trainer.step(data.shape[0])
                # update metric at last.
                metric.update([label], [output])

                if i % log_interval == 0 and i > 0:
                    name, acc = metric.get()
                    print('[Epoch %d Batch %d] Training: %s=%f' % (epoch, i, name, acc))

            name, acc = metric.get()
            print('[Epoch %d] Training: %s=%f' % (epoch, name, acc))

            name, val_acc = self._test(
                model=self.model,
                val_data=val_loader,
                ctx=self.ctx)
            print('[Epoch %d] Validation: %s=%f' % (epoch, name, val_acc))

        if self.saver.log_filename:
            self.logger.removeHandler(fh)
            fh.close()

        best_value = 0.0

        return best_value

    @staticmethod
    def _test(model,
              val_data,
              ctx):
        metric = mx.metric.Accuracy()
        for data, label in val_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = model(data)
            metric.update([label], [output])

        return metric.get()

