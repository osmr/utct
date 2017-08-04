import os
import logging

import tensorflow as tf

from .evaluator import Evaluator


class Estimator(object):

    @staticmethod
    def estimate(model,
                 data_source,
                 checkpoint_path,
                 **kwargs):

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        data = data_source(**kwargs)
        with tf.get_default_graph().as_default():
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            session = tf.Session(config=session_config)

            try:
                session.run([tf.global_variables_initializer(),
                             tf.local_variables_initializer()])
            except Exception:
                session.run(tf.initialize_all_variables())

            net, loss, metric = model(
                is_training=False,
                **kwargs)

            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), checkpoint_path))
            restorer_trainvars = tf.train.Saver(var_list=tf.trainable_variables())
            restorer_trainvars.restore(session, checkpoint_path)

            predictor = Evaluator([net], session=session)
            train_score = Estimator._evaluate(
                predictor,
                X=data["X_inputs"],
                Y=data["Y_targets"],
                batch_size=data_source.batch_size,
                metric=metric)
            val_score = Estimator._evaluate(
                predictor,
                X=data["validation_set"][0],
                Y=data["validation_set"][1],
                batch_size=data_source.batch_size,
                metric=metric)

            logger.info("Train score: {}".format(train_score))
            logger.info("Validation score: {}".format(val_score))

    @staticmethod
    def _evaluate(predictor, X, Y, batch_size, metric):
        inputs = tf.get_collection(tf.GraphKeys.INPUTS)
        targets = tf.get_collection(tf.GraphKeys.TARGETS)
        feed_dict = {inputs[0]: X, targets[0]: Y}
        return predictor.evaluate(
            feed_dict=feed_dict,
            ops=[metric],
            batch_size=batch_size)
