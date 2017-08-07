import os
import logging

from .best_state_saver import BestStateSaverCallback, TrainControllerStopException
from utct.common.trainer_template import TrainerTemplate
from .trainer import TrainOp
from .dnn import DNN

import tensorflow as tf
#import tensorflow.contrib.slim as slim


class Trainer1(TrainerTemplate):
    """
    Class, which provides training process under TensorFlow framework.
    """

    def _hyper_train_target_sub(self, **kwargs):
        """
        Actual training procedure for specific set of hyper parameters.
        """

        if self.saver.log_filename:
            fh = logging.FileHandler(self.saver.log_filename)
            self.logger.addHandler(fh)

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        tf.GraphKeys.GRAPH_CONFIG = 'graph_config'
        tf.add_to_collection(tf.GraphKeys.GRAPH_CONFIG, session_config)

        data = self.data_source(**kwargs)

        net, loss, metric = self.model(**kwargs)
        tr_op = TrainOp(loss=loss,
                        optimizer=self.optimizer(),
                        metric=metric,
                        trainable_vars=tf.trainable_variables(),
                        batch_size=self.data_source.batch_size,
                        shuffle=True,
                        step_tensor=None,
                        validation_monitors=None,
                        validation_batch_size=None,
                        name=None)
        tf.GraphKeys.TRAIN_OPS = 'trainops'
        tf.add_to_collection(tf.GraphKeys.TRAIN_OPS, tr_op)

        data_augmentation = self.data_source.img_aug
        tf.GraphKeys.DATA_AUG = 'data_augmentation'
        tf.add_to_collection(tf.GraphKeys.DATA_AUG, data_augmentation)

        model = DNN(network=net,
                    tensorboard_verbose=0,
                    tensorboard_dir=str(self.saver.log_dirname),
                    checkpoint_path=os.path.join(self.saver.last_checkpoints_dirname, self.saver.model_filename_prefix),
                    max_checkpoints=2)

        bss_callback = BestStateSaverCallback(
            session=model.session,
            best_snapshot_path=os.path.join(self.saver.best_checkpoints_dirname, self.saver.model_filename_prefix),
            best_val_accuracy=0.0,
            epoch_tail=self.epoch_tail)

        try:
            model.fit(n_epoch=self.num_epoch,
                      show_metric=True,
                      batch_size=self.data_source.batch_size,
                      shuffle=True,
                      snapshot_epoch=True,
                      run_id=self.saver.project_name,
                      callbacks=bss_callback,
                      **data)
        except TrainControllerStopException as e:
            model.trainer.summ_writer.close()
            self.logger.info(e)

        self.logger.info("Best validation accuracy: {:.4f} (at epoch {})".format(
            bss_callback.best_val_accuracy, bss_callback.best_epoch))

        if self.saver.log_filename:
            self.logger.removeHandler(fh)
            fh.close()

        tf.reset_default_graph()
        model.session.close()

        return bss_callback.best_val_accuracy
