import tensorflow as tf
import tflearn
from tflearn.utils import fix_saver

from utct.common.overfitting_detector import OverfittingDetector
from utct.common.train_controller_stop_exception import TrainControllerStopException


class BestStateSaverCallback(tflearn.callbacks.Callback):
    def __init__(self,
                 session,
                 best_snapshot_path,
                 best_val_accuracy=1e5,
                 bigger=True,
                 max_checkpoints=2,
                 epoch_tail=None,
                 min_num_epoch=0,
                 lowess_factor=0.5,
                 overfitting_tol=1e-5):
        self.best_snapshot_path = best_snapshot_path
        self.best_val_accuracy = best_val_accuracy
        self.bigger = bigger
        self.best_epoch = -1
        self.session = session
        self.snapshot_path = None
        with tf.get_default_graph().as_default():
            self.saver = tf.train.Saver(
                max_to_keep=max_checkpoints,
                name="best_model_saver")

        self.of_detector = OverfittingDetector(
            epoch_tail,
            min_num_epoch,
            bigger,
            lowess_factor,
            overfitting_tol)

    def on_epoch_end(self, training_state):
        is_better = self.check_best(training_state)
        if self.of_detector.check(training_state.val_acc, is_better):
            raise TrainControllerStopException("Reached the maximum number of epochs,\n\teval_metric={},\n\ty_tail={},\n\ty_smooth={}".format(training_state.val_acc, self.of_detector.y_tail, self.of_detector.y_smooth))

    def on_batch_end(self, training_state, snapshot=False):
        #self.check_best(training_state)
        pass

    def check_best(self, training_state):
        if None not in (self.best_snapshot_path, self.best_val_accuracy, training_state.val_acc):
            if training_state.val_acc == self.best_val_accuracy:
                return False
            if (training_state.val_acc > self.best_val_accuracy) == self.bigger:
                self.best_val_accuracy = training_state.val_acc
                self.best_epoch = training_state.epoch
                self.save_best(int(10000 * round(training_state.val_acc, 4)), training_state.epoch)
                return True
        return False

    def save_best(self, val_accuracy, epoch):
        if self.best_snapshot_path:
            self.snapshot_path = self.best_snapshot_path + "-" + str(val_accuracy) + "-" + str(epoch)
            #self.saver.save(self.session, snapshot_path, global_step=None)
            self.save(self.snapshot_path)

    def save(self, model_file, global_step=None):
        obj_lists = fix_saver()
        self.saver.save(self.session, model_file, global_step=global_step)
        fix_saver(obj_lists)


class TrainControllerStopException(Exception):
    pass
