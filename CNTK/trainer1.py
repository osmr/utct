import logging

from utct.common.trainer_template import TrainerTemplate

from cntk import Trainer
from cntk.ops import cross_entropy_with_softmax, classification_error
from cntk.utils import ProgressPrinter


class Trainer1(TrainerTemplate):
    """
    Class, which provides training process under CNTK framework.
    """

    def _hyper_train_target_sub(self, **kwargs):
        """
        Actual training procedure for specific set of hyper parameters.
        """

        if self.saver.log_filename:
            fh = logging.FileHandler(self.saver.log_filename)
            self.logger.addHandler(fh)

        self.logger.info("Training with parameters: {}".format(kwargs))

        X_train, Y_train, X_val, Y_val = self.data_source(**kwargs)

        input_var, label_var, output = self.model(**kwargs)

        loss = cross_entropy_with_softmax(output, label_var)
        label_error = classification_error(output, label_var)

        learner = self.optimizer(
            parameters=output.parameters,
            momentum=0.9,
            **kwargs)

        progress_printer = ProgressPrinter(tag='Training', num_epochs=self.num_epoch)
        trainer = Trainer(output, (loss, label_error), [learner], [progress_printer])

        # input_map = {
        #     input_var: reader_train.streams.features,
        #     label_var: reader_train.streams.labels
        # }

        num_minibatches_to_train = X_train.shape[0] / self.data_source.batch_size
        for i in range(0, int(num_minibatches_to_train)):
            features = X_train[:self.data_source.batch_size]
            labels = Y_train[:self.data_source.batch_size]
            trainer.train_minibatch({input_var: features, label_var: labels})


        if self.saver.log_filename:
            self.logger.removeHandler(fh)
            fh.close()

        best_value = 0.0

        return best_value
