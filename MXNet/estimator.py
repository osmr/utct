import logging
import mxnet as mx


class Estimator(object):

    @staticmethod
    def estimate(data_source,
                 checkpoint_path,
                 checkpoint_epoch,
                 ctx,
                 **kwargs):

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        mod = mx.mod.Module.load(
            prefix=checkpoint_path,
            epoch=checkpoint_epoch,
            logger=logging,
            context=ctx)

        train_iter, val_iter = data_source(shuffle=False, **kwargs)

        mod.bind(
            data_shapes=train_iter.provide_data,
            label_shapes=train_iter.provide_label,
            for_training=False)

        acc_metric = mx.metric.Accuracy()
        logger.info("Train score: {}".format(mod.score(train_iter, acc_metric)))
        logger.info("Validation score: {}".format(mod.score(val_iter, acc_metric)))
