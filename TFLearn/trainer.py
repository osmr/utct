import os
import logging

from .best_state_saver import BestStateSaverCallback, TrainControllerStopException
from . import train_op_cp
from utct.common.trainer_template import TrainerTemplate

import tensorflow as tf
import tflearn


class Trainer(TrainerTemplate):
    """
    Class, which provides training process under TFLearn framework.
    """

    def _hyper_train_target_sub(self, **kwargs):
        """
        Actual training procedure for specific set of hyper parameters.
        """

        if self.saver.log_filename:
            fh = logging.FileHandler(self.saver.log_filename)
            self.logger.addHandler(fh)

        config = tflearn.init_graph()
        config.gpu_options.allow_growth = True

        data = self.data_source(**kwargs)

        net = self.model(
            self.optimizer(**kwargs),
            self.data_source,
            **kwargs)
        model = tflearn.DNN(network=net,
                            tensorboard_verbose=(kwargs['tensorboard_verbose'] if 'tensorboard_verbose' in kwargs else 0),
                            tensorboard_dir=str(self.saver.log_dirname),
                            checkpoint_path=os.path.join(self.saver.last_checkpoints_dirname, self.saver.model_filename_prefix),
                            max_checkpoints=2)

        if self.data_source.rewrite_data_aug:
            train_op_cp.replace_train_op_initialize_fit_cp(model)

        bss_callback = BestStateSaverCallback(
            session=model.session,
            best_snapshot_path=os.path.join(self.saver.best_checkpoints_dirname, self.saver.model_filename_prefix),
            best_val_accuracy=(kwargs['bss_best_val_accuracy'] if 'bss_best_val_accuracy' in kwargs else 0.0),
            bigger=(kwargs['bss_bigger'] if 'bss_bigger' in kwargs else True),
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
