import logging

from .train_controller import TrainController, TrainControllerStopException
from utct.common.trainer_template import TrainerTemplate

import mxnet as mx


class Trainer(TrainerTemplate):
    """
    Class, which provides training process under MXNet framework.

    Parameters:
    ----------
    model : object
        instance of Model class with graph of CNN
    optimizer : object
        instance of Optimizer class with CNN optimizer
    data_source : object
        instance of DataSource class with training/validation iterators
    saver : object
        instance of Saver class with information about stored files
    ctx : object
        instance of MXNet context
    tc_bigger : list of bool
        must quality indexes increase
    eval_metric : str
        name of MXNet evaluation metric
    """

    def __init__(self,
                 model,
                 optimizer,
                 data_source,
                 saver,
                 ctx,
                 tc_bigger=[True],
                 eval_metric='acc',
                 **kwargs):
        super(Trainer, self).__init__(
            model,
            optimizer,
            data_source,
            saver)
        self.ctx = ctx
        self.tc_bigger = tc_bigger
        self.eval_metric = eval_metric

    def train(self,
              num_epoch,
              epoch_tail,
              **kwargs):
        """
        A point of entry for single training procedure.

        Parameters:
        ----------
        num_epoch : int
            maximal number of training epochs
        epoch_tail : int
            number of epochs for overfitting detection
        """
        self._prepare_train()
        super(Trainer, self).train(num_epoch, epoch_tail, **kwargs)

    def hyper_train(self,
                    num_epoch,
                    epoch_tail,
                    bo_num_iter,
                    bo_kappa,
                    bo_min_rand_num,
                    bo_results_filename,
                    synch_file_list=[],
                    sync_period=5):
        """
        A point of entry for multiple training procedure.

        Parameters:
        ----------
        num_epoch : int
            maximal number of training epochs
        epoch_tail : int
            number of epochs for overfitting detection
        bo_num_iter : int
            number of attempts for bayesian optimization
        bo_kappa : float
            kappa parameter for bayesian optimization
        bo_min_rand_num : int
            minimal number of random attempts for overfitting detection
        bo_results_filename : str
            name of file for results of bayesian optimization
        synch_file_list : str
            name of file for synchronization of several instances of hyper optimizers
        sync_period : int
            number of attempts between synchronizations of several instances of hyper optimizers
        """
        self._prepare_train()
        super(Trainer, self).hyper_train(
            num_epoch,
            epoch_tail,
            bo_num_iter,
            bo_kappa,
            bo_min_rand_num,
            bo_results_filename,
            synch_file_list,
            sync_period)

    def _hyper_train_target_sub(self, **kwargs):
        """
        Calling single training procedure for specific hyper parameters from hyper optimizer.
        """

        if self.saver.log_filename:
            fh = logging.FileHandler(self.saver.log_filename)
            self.logger.addHandler(fh)

        self.logger.info("Training with parameters: {}".format(kwargs))

        train_controller = TrainController(
            checkpoints_filename_prefix=self.saver.model_filename_prefix,
            last_checkpoints_dirname=self.saver.last_checkpoints_dirname,
            best_checkpoints_dirname=self.saver.best_checkpoints_dirname,
            bigger=self.tc_bigger,
            score_log_filename=self.saver.score_log_filename,
            epoch_tail=self.epoch_tail)
        if self.iter is not None:
            train_controller.score_log_attempt = self.iter

        train_iter, val_iter = self.data_source(**kwargs)

        mod = mx.mod.Module(
            symbol=self.model(**kwargs),
            logger=self.logger,
            context=self.ctx)

        optimizer = self.optimizer(**kwargs)

        batch_end_callback = [
            mx.callback.Speedometer(self.data_source.batch_size, 100),
            train_controller.get_batch_end_callback()]

        try:
            mod.fit(
                train_data=train_iter,
                eval_data=val_iter,
                eval_metric=self.eval_metric,
                epoch_end_callback=train_controller.get_epoch_end_callback(),
                batch_end_callback=batch_end_callback,
                eval_end_callback=train_controller.get_score_end_callback(),
                optimizer=optimizer,
                initializer=mx.init.Xavier(),
                arg_params=self.arg_params,
                aux_params=self.aux_params,
                #force_rebind=True,
                #force_init=True,
                begin_epoch=self.begin_epoch,
                num_epoch=self.num_epoch)
        except TrainControllerStopException as e:
            self.logger.info(e)

        train_controller.log_best_results(logger=self.logger)

        if self.saver.log_filename:
            self.logger.removeHandler(fh)
            fh.close()

        best_value = (train_controller.best_eval_metric_values[-1][0] if len(
            train_controller.best_eval_metric_values) > 0 else 0.0)

        del train_controller
        del mod

        return best_value

    def _prepare_train(self):
        """
        Initialization of internal fields before single optimization.
        """
        self.begin_epoch = 0
        self.arg_params = None
        self.aux_params = None
