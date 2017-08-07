import os
import shutil

import mxnet as mx
import numpy as np

from utct.common.overfitting_detector import OverfittingDetector
from utct.common.train_controller_stop_exception import TrainControllerStopException


class TrainController(object):
    """
    Train controller, that does the following:
    1. save several the last model checkpoints, for disaster recovery,
    2. save several the best model checkpoints, to prevent overfitting,
    3. save pure evaluation metric values to log-file for observer,
    4. detect overfitting and break training.

    Parameters:
    ----------
    checkpoints_filename_prefix : str
        preffix for checkpoint files without parent dir
    last_checkpoints_dirname : str
        directory name for storing the last checkpoint files
    last_checkpoint_file_count : int
        count of the last checkpoint files
    best_checkpoints_dirname : str or None
        directory name for storing the best checkpoint files
    best_checkpoint_file_count : int
        count of the best checkpoint files
    bigger : list of bool
        Should be bigger for each value of evaluation metric values
    mask : list of bool or None
        evaluation metric values that should be taken into account
    score_log_filename : str
        file name for score log storing (full path)
    score_log_attempt : int
        number of current attempt, if we write data for various optimizations
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
                 checkpoints_filename_prefix="model",
                 last_checkpoints_dirname="",
                 last_checkpoint_file_count=2,
                 best_checkpoints_dirname=None,
                 best_checkpoint_file_count=2,
                 bigger=[True],
                 mask=None,
                 score_log_filename=None,
                 score_log_attempt=1,
                 epoch_tail=None,
                 min_num_epoch=0,
                 lowess_factor=0.5,
                 overfitting_tol=1e-5):
        assert (best_checkpoints_dirname is None) or\
            (last_checkpoints_dirname != best_checkpoints_dirname)
        if not os.path.exists(last_checkpoints_dirname):
            os.makedirs(last_checkpoints_dirname)
        if (best_checkpoints_dirname is not None) and\
                (not os.path.exists(best_checkpoints_dirname)):
            os.makedirs(best_checkpoints_dirname)

        self.checkpoints_filename_prefix = checkpoints_filename_prefix
        self.last_checkpoints_prefix = os.path.join(
            last_checkpoints_dirname, checkpoints_filename_prefix)
        self.last_checkpoint_file_count = last_checkpoint_file_count
        self.best_checkpoints_prefix = os.path.join(best_checkpoints_dirname, checkpoints_filename_prefix)
        assert best_checkpoint_file_count > 0
        self.best_checkpoint_file_count = best_checkpoint_file_count
        assert isinstance(bigger, list)
        self.bigger = np.array(bigger)
        if mask is None:
            self.mask = np.ones_like(self.bigger)
        else:
            assert isinstance(mask, list)
            assert len(mask) == len(bigger)
            self.mask = np.array(mask)
        if score_log_filename is not None:
            self.score_log_file_exist = os.path.exists(score_log_filename) and\
                                        os.path.getsize(score_log_filename) > 0
            self.score_log_file = open(score_log_filename, "a")
        else:
            self.score_log_file = None

        self.score_log_attempt = score_log_attempt

        self.of_detector = OverfittingDetector(
            epoch_tail,
            min_num_epoch,
            bigger[0],
            lowess_factor,
            overfitting_tol)
        self.last_checkpoint_epochs = []
        self.best_eval_metric_epochs = []
        self.best_eval_metric_values = []
        self.last_batch_values = None
        self.last_score_values = None

    def __del__(self):
        """
        Releasing resources.
        """
        if self.score_log_file is not None:
            self.score_log_file.close()

    def get_epoch_end_callback(self):
        """
        Get link to internal method _epoch_end_callback.
        """
        return self._epoch_end_callback

    def get_score_end_callback(self):
        """
        Get link to internal method _score_end_callback.
        """
        return self._score_end_callback

    def get_batch_end_callback(self):
        """
        Get link to internal method _batch_end_callback.
        """
        return self._batch_end_callback

    def log_best_results(self, logger=None):
        """
        Logging standard results of CNN training.

        Parameters:
        ----------
        logger : object
            instance of Logger
        """
        if logger is None:
            return
        logger.info("Best eval metric values:")
        for best_epoch, best_values in zip(self.best_eval_metric_epochs, self.best_eval_metric_values):
            logger.info("Epoch[{}]: {}".format(best_epoch, best_values))

    @staticmethod
    def _get_checkpoint_params_filename(checkpoints_prefix, epoch):
        return "{0}-{1:04d}.params".format(checkpoints_prefix, epoch)

    @staticmethod
    def _get_checkpoint_symbol_filename(checkpoints_prefix):
        return "{0}-symbol.json".format(checkpoints_prefix)

    def _get_last_checkpoint_params_filename(self, epoch):
        return self._get_checkpoint_params_filename(self.last_checkpoints_prefix, epoch)

    def _get_best_checkpoint_params_filename(self, epoch):
        return self._get_checkpoint_params_filename(self.best_checkpoints_prefix, epoch)

    def _get_last_checkpoint_symbol_filename(self):
        return self._get_checkpoint_symbol_filename(self.last_checkpoints_prefix)

    def _get_best_checkpoint_symbol_filename(self):
        return self._get_checkpoint_symbol_filename(self.best_checkpoints_prefix)

    def _epoch_end_callback(self, epoch, sym, arg, aux):
        mx.model.save_checkpoint(self.last_checkpoints_prefix, epoch + 1, sym, arg, aux)
        self.last_checkpoint_epochs.append(epoch + 1)
        if len(self.last_checkpoint_epochs) > self.last_checkpoint_file_count:
            removed_checkpoint_filename = self._get_last_checkpoint_params_filename(
                self.last_checkpoint_epochs[0])
            if os.path.exists(removed_checkpoint_filename):
                os.remove(removed_checkpoint_filename)
            del self.last_checkpoint_epochs[0]

    def _is_better(self, values):
        assert len(self.best_eval_metric_values) > 0
        assert len(self.best_eval_metric_values[-1]) == len(values)
        assert len(self.bigger) == len(values)
        if np.any(values == self.best_eval_metric_values[-1]):
            return False
        value_bigger = values > self.best_eval_metric_values[-1]
        return np.array_equal(value_bigger[self.mask], self.bigger[self.mask])

    @staticmethod
    def _get_eval_metric_values(eval_metric):
        values = eval_metric.get()[1]
        if not isinstance(values, list):
            values = [values]
        return np.array(values)

    def _batch_end_callback(self, param):
        if param.eval_metric is None:
            return
        values = self._get_eval_metric_values(param.eval_metric)
        self.last_batch_values = values.copy()

    def _log_score_values(self, epoch, values):
        if self.score_log_file is not None:
            score_log_values = [self.score_log_attempt, epoch] + list(values)
            if self.last_batch_values is not None:
                score_log_values += list(self.last_batch_values)
            self.score_log_file.write("\t".join(list(map(str, score_log_values)))+"\n")
            self.score_log_file.flush()

    def _score_end_callback(self, param):
        if param.eval_metric is None:
            return
        values = self._get_eval_metric_values(param.eval_metric)
        if len(self.best_eval_metric_values) == 0:
            #self.best_eval_metric_epochs.append(param.epoch + 1)
            #self.best_eval_metric_values.append(values.copy())
            if self.score_log_file is not None and not self.score_log_file_exist:
                metric_names = param.eval_metric.get()[0]
                if not isinstance(metric_names, list):
                    metric_names = [metric_names]
                titles = ["Attempt", "Epoch"] + list(map(lambda x: "Val." + x, metric_names))
                if self.last_batch_values is not None:
                    titles += list(map(lambda x: "Train." + x, metric_names))
                self.score_log_file.write("\t".join(titles)+"\n")
            is_better = True
        else:
            is_better = self._is_better(values)
        self._log_score_values(param.epoch + 1, values)
        if is_better:
            assert len(self.last_checkpoint_epochs) > 0
            best_epoch = self.last_checkpoint_epochs[-1]
            shutil.copy(self._get_last_checkpoint_params_filename(best_epoch),
                        self._get_best_checkpoint_params_filename(best_epoch))
            if len(self.best_eval_metric_epochs) == 0:
                shutil.copy(self._get_last_checkpoint_symbol_filename(),
                            self._get_best_checkpoint_symbol_filename())
            self.best_eval_metric_epochs.append(best_epoch)
            self.best_eval_metric_values.append(values.copy())
            if len(self.best_eval_metric_epochs) > self.best_checkpoint_file_count:
                removed_best_checkpoint_filename = self._get_best_checkpoint_params_filename(self.best_eval_metric_epochs[0])
                if os.path.exists(removed_best_checkpoint_filename):
                    os.remove(removed_best_checkpoint_filename)
                del self.best_eval_metric_epochs[0]
                del self.best_eval_metric_values[0]
        self.last_score_values = values.copy()
        if self.of_detector.check(values[0], is_better):
            raise TrainControllerStopException("Reached the maximum number of epochs,\n\teval_metric={},\n\ty_tail={},\n\ty_smooth={}".format(values, self.of_detector.y_tail, self.of_detector.y_smooth))
