import tensorflow as tf

from .callbacks import Callback
from utct.common.overfitting_detector import OverfittingDetector
from utct.common.train_controller_stop_exception import TrainControllerStopException


class BestStateSaverCallback(Callback):
    """
    Train controller, that does the following:
    1. save several the last model checkpoints, for disaster recovery,
    2. save several the best model checkpoints, to prevent overfitting,
    3. save pure evaluation metric values to log-file for observer,
    4. detect overfitting and break training.

    Parameters:
    ----------
    session : object
        insrtance of TensorFlow Session object
    best_snapshot_path : str
        file path for storing the best checkpoint files
    best_val_accuracy : float
        Initial value of validation accuracy
    bigger : bool
        Should be bigger for each value of evaluation metric values
    max_checkpoints : int
        count of the checkpoint files (best/last)
    epoch_tail : int or None
        number of epochs for analysing of convergence
    min_num_epoch : int or None
        pure minimum number of epochs that should be without extra logic
    lowess_factor : float
        factor for LOWESS smoothing of tail
    overfitting_tol : float
        TOL for overfitting detection
    """

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
        l = tf.get_collection_ref("summary_tags")
        l4 = tf.get_collection_ref(tf.GraphKeys.GRAPH_CONFIG)
        l_stags = list(l)
        l4_stags = list(l4)
        del l[:]
        del l4[:]

        l1 = tf.get_collection_ref(tf.GraphKeys.DATA_PREP)
        l2 = tf.get_collection_ref(tf.GraphKeys.DATA_AUG)
        l1_dtags = list(l1)
        l2_dtags = list(l2)
        del l1[:]
        del l2[:]

        l3 = tf.get_collection_ref(tf.GraphKeys.EXCL_RESTORE_VARS)
        l3_tags = list(l3)
        del l3[:]

        self.saver.save(self.session, model_file, global_step=global_step)

        for t in l_stags:
            tf.add_to_collection("summary_tags", t)
        for t in l4_stags:
            tf.add_to_collection(tf.GraphKeys.GRAPH_CONFIG, t)
        for t in l1_dtags:
            tf.add_to_collection(tf.GraphKeys.DATA_PREP, t)
        for t in l2_dtags:
            tf.add_to_collection(tf.GraphKeys.DATA_AUG, t)
        for t in l3_tags:
            tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, t)


class TrainControllerStopException(Exception):
    """
    An exception to interrupt the training process (TODO: change to common exception!).
    """
    pass
