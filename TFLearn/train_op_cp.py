import numpy as np
from types import MethodType
from tflearn import utils
from .feed_dict_flow_cp import FeedDictFlowCp

"""
This module contains wrappers for some TFLearn's methods (for some types of augmentation).
"""


def train_op_initialize_fit_cp(self, feed_dict, val_feed_dict, dprep_dict, daug_dict,
                               show_metric, summ_writer, coord):
    """ initialize_fit.
    Initialize data for feeding the training process. It is meant to
    be used by `Trainer` before starting to fit data.
    Arguments:
        feed_dict: `dict`. The data dictionary to feed.
        val_feed_dict: `dict` or `float`. The validation data dictionary to
            feed or validation split.
        dprep_dict: `dict`. Data Preprocessing dict (with placeholder as
            key and corresponding `DataPreprocessing` object as value).
        daug_dict: `dict`. Data Augmentation dict (with placeholder as
            key and corresponding `DataAugmentation` object as value).
        show_metric: `bool`. If True, display accuracy at every step.
        summ_writer: `SummaryWriter`. The summary writer to use for
            Tensorboard logging.
    """
    self.summary_writer = summ_writer
    self.feed_dict = feed_dict
    self.val_feed_dict = val_feed_dict
    self.n_train_samples = len(utils.get_dict_first_element(feed_dict))

    self.index_array = np.arange(self.n_train_samples)
    self.n_val_samples = 0
    # Validation Split
    # TODO: Optional per key validation split
    if isinstance(val_feed_dict, float):
        split_at = int(self.n_train_samples * (1 - val_feed_dict))
        # Shuffle Data
        np.random.shuffle(self.index_array)
        self.val_index_array = self.index_array[split_at:]
        self.index_array = self.index_array[:split_at]
        self.n_train_samples = len(self.index_array)
        self.n_val_samples = len(self.val_index_array)
        val_feed_dict = feed_dict
    elif val_feed_dict is not None:
        self.val_index_array = None
        self.n_val_samples = len(utils.get_dict_first_element(val_feed_dict))

    if dprep_dict:
        for k in dprep_dict:
            assert feed_dict[k] is not None, \
                "Unknown DataPreprocessing dict key!"
            dprep_dict[k].initialize(feed_dict[k], self.session)
    self.train_dflow = FeedDictFlowCp(feed_dict, coord,
                                      continuous=True,
                                      batch_size=self.batch_size,
                                      dprep_dict=dprep_dict,
                                      daug_dict=daug_dict,
                                      index_array=self.index_array,
                                      num_threads=1,
                                      shuffle=self.shuffle)

    self.n_batches = len(self.train_dflow.batches)
    self.train_dflow.start()
    # TODO: Optimize data_flow to not start/restart threads (cost time)
    # every time testing
    if val_feed_dict:
        self.test_dflow = FeedDictFlowCp(val_feed_dict, coord,
                                         batch_size=self.validation_batch_size,
                                         dprep_dict=dprep_dict,
                                         daug_dict=None,
                                         index_array=self.val_index_array,
                                         num_threads=1)

    self.create_testing_summaries(show_metric, self.metric_summ_name,
                                  val_feed_dict)


def replace_train_op_initialize_fit_cp(model):
    for t in model.trainer.train_ops:
        t.initialize_fit = MethodType(train_op_initialize_fit_cp, t)

