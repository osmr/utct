import logging
import mxnet as mx


class Estimator(object):
    """
    Class, which provides recalculation of quality indexes (for classifier!).
    """

    @staticmethod
    def estimate(data_source,
                 checkpoint_path,
                 checkpoint_epoch,
                 ctx,
                 **kwargs):
        """
        Recalculating quality indexes.

        Parameters:
        ----------
        data_source : object
            instance of DataSource class with training/validation iterators
        checkpoint_path : str
            path to checkpoint file with the prefix
        checkpoint_epoch : int
            number of epoch for the checkpoint file
        ctx : object
            instance of MXNet context
        """

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
