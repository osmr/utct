import logging
import time
import numpy as np

import theano
import theano.tensor as T
import lasagne

from utct.common.trainer_template import TrainerTemplate


class Trainer(TrainerTemplate):

    def _hyper_train_target_sub(self, **kwargs):

        if self.saver.log_filename:
            fh = logging.FileHandler(self.saver.log_filename)
            self.logger.addHandler(fh)

        self.logger.info("Training with parameters: {}".format(kwargs))

        X_train, y_train, X_val, y_val = self.data_source(**kwargs)

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        network = self.model(input_var=input_var, **kwargs)

        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = self.optimizer(loss, params, **kwargs)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        for epoch in range(self.num_epoch):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in Trainer._iterate_minibatches(X_train, y_train, batch_size=self.data_source.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in Trainer._iterate_minibatches(X_val, y_val, batch_size=self.data_source.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            self.logger.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epoch, time.time() - start_time))
            self.logger.info("\ttraining loss:\t\t{:.6f}".format(train_err / train_batches))
            self.logger.info("\tvalidation loss:\t\t{:.6f}".format(val_err / val_batches))
            self.logger.info("\tvalidation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        if self.saver.log_filename:
            self.logger.removeHandler(fh)
            fh.close()

        best_value = 0.0

        return best_value

    @staticmethod
    def _iterate_minibatches(inputs, targets, batch_size, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]
