import logging
import tflearn


class Estimator(object):
    """
    Class, which provides recalculation of quality indexes (for classifier!).
    """

    @staticmethod
    def estimate(model,
                 optimizer,
                 data_source,
                 checkpoint_path,
                 **kwargs):
        """
        Recalculating quality indexes.

        Parameters:
        ----------
        model : object
            instance of Model class with graph of CNN
        optimizer : object
            instance of Optimizer class with CNN optimizer
        data_source : object
            instance of DataSource class with training/validation iterators
        checkpoint_path : str
            path to checkpoint file
        """

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        config = tflearn.init_graph()
        config.gpu_options.allow_growth = True

        net = model(
            optimizer(),
            data_augmentation=None,
            is_training=False,
            **kwargs)
        model = tflearn.DNN(network=net,
                            tensorboard_verbose=0,
                            checkpoint_path=checkpoint_path,
                            max_checkpoints=2)
        model.load(model_file=checkpoint_path, weights_only=True)

        data = data_source(**kwargs)
        train_score = model.evaluate(
            X=data["X_inputs"],
            Y=data["Y_targets"],
            batch_size=data_source.batch_size)
        val_score = model.evaluate(
            X=data["validation_set"][0],
            Y=data["validation_set"][1],
            batch_size=data_source.batch_size)

        logger.info("Train score: {}".format(train_score))
        logger.info("Validation score: {}".format(val_score))
