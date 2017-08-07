import os
import logging
import time
import pandas as pd

from bayes_opt import BayesianOptimization


class TrainerTemplate(object):
    """
    Base class for Trainer (class, which provides training process).

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
    """

    def __init__(self,
                 model,
                 optimizer,
                 data_source,
                 saver):
        self.model = model
        self.optimizer = optimizer
        self.data_source = data_source
        self.saver = saver

        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.iter = None

    def _hyper_train_target_sub(self, **kwargs):
        """
        Actual training procedure for specific set of hyper parameters.
        """
        raise NotImplementedError()

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
        self.num_epoch = num_epoch
        self.epoch_tail = epoch_tail
        self.saver.update_log_filename(self.saver.model_filename_prefix + ".log")
        self._hyper_train_target_sub(**kwargs)

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

        param_bounds = dict()
        param_bounds.update(self.model.param_bounds)
        param_bounds.update(self.optimizer.param_bounds)
        param_bounds.update(self.data_source.param_bounds)

        self.num_epoch = num_epoch
        self.epoch_tail = epoch_tail

        hyper_log_file_exist = os.path.exists(self.saver.hyper_log_filename) and (os.path.getsize(self.saver.hyper_log_filename) > 0)

        self.hyper_log_file = open(self.saver.hyper_log_filename, "a")
        bo = BayesianOptimization(self._hyper_train_target, param_bounds, verbose=1)
        bo_init_points = max(bo_min_rand_num, len(param_bounds.keys()))

        if hyper_log_file_exist:
            try:
                df = pd.read_csv(self.saver.hyper_log_filename, sep="\t")

                if df.isnull().values.any():
                    raise Exception()

                if len(df.index) > 0:
                    bo.initialize_df(df)

                    self.iter = df['iter'].max()
                    global_best_row_id = df['target'].idxmax()
                    self.global_best_value = df['target'].loc[global_best_row_id]
                    self.global_best_iter = df['iter'].loc[global_best_row_id]

                    bo_init_points = max(0, bo_init_points - self.iter)
                else:
                    self._init_iter()
            except:
                raise Exception("Error: Hyperparameter optimization's file can't be properly read: {}".format(self.saver.hyper_log_filename))
        else:
            self._init_iter()
            self.hyper_log_file.write("\t".join(["iter", "target", "time"] + sorted(param_bounds)) + "\n")
            self.hyper_log_file.flush()

        try:
            if len(synch_file_list) == 0:
                bo.maximize(
                    init_points=bo_init_points,
                    n_iter=bo_num_iter,
                    kappa=bo_kappa)
            else:
                if bo_init_points > 0:
                    bo.maximize(
                        init_points=bo_init_points,
                        n_iter=0,
                        kappa=bo_kappa)
                for i in range(0, bo_num_iter, sync_period):
                    df = self._update_df(self.saver.hyper_log_filename, df=None)
                    for synch_filename in synch_file_list:
                        df = self._update_df(synch_filename, df)
                    bo = BayesianOptimization(
                        self._hyper_train_target,
                        param_bounds,
                        verbose=1)
                    bo.initialize_df(df)
                    bo.maximize(
                        init_points=0,
                        n_iter=sync_period,
                        kappa=bo_kappa)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: saving...")
        bo.points_to_csv(os.path.join(self.saver.project_dirname, bo_results_filename))

        self.hyper_log_file.close()

        print(bo.res['max'])
        print("Results: global_best_value={}, global_best_iter={}".format(self.global_best_value, self.global_best_iter))

    def _hyper_train_target(self, **kwargs):
        """
        Calling single training procedure for specific hyper parameters from hyper optimizer.
        """

        self.iter += 1

        model_filename_prefix = "{0}_{1:04d}".format(self.saver.model_filename_prefix, self.iter)
        self.saver.update_log_filename(model_filename_prefix + ".log")

        start_time = time.time()

        best_value = self._hyper_train_target_sub(**kwargs)

        if best_value > self.global_best_value:
            self.global_best_value = best_value
            self.global_best_iter = self.iter

        time_eval = time.time() - start_time
        self.hyper_log_file.write("\t".join(
            list(map(str, [self.iter, best_value, time_eval] +
                [kwargs[key] for key in sorted(kwargs)]))) + "\n")
        self.hyper_log_file.flush()

        return best_value

    def _init_iter(self):
        """
        Initialization of internal fields before hyper optimization.
        """
        self.iter = 0
        self.global_best_value = -1.0
        self.global_best_iter = 0

    def _update_df(self,
                   filename,
                   df):
        """
        Saving hyper optimizing score and synchronizing scores between several hyper optimizers.

        Parameters:
        ----------
        filename : str
            name of file for score saving
        df : object
            table with scores
        """

        if not os.path.exists(filename):
            filename = os.path.join(self.saver.work_dirname, filename)
        if not (os.path.exists(filename) and (os.path.getsize(filename) > 0)):
            return df
        extra_df = pd.read_csv(filename, sep="\t")
        if extra_df.isnull().values.any():
            return df
        print("Synchronizing: {} records are extracted from file '{}'".format(
            len(extra_df.index), filename))
        if df is None:
            df = extra_df
            return df
        df = df.append(extra_df, ignore_index=True)
        return df
