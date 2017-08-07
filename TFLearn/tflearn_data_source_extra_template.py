class TflearnDataSourceExtraTemplate(object):
    """
    Base class for TFLearn's DataSource (if we use wrapping).

    Parameters:
    ----------
    rewrite_data_aug : bool
        use wrapper for data augmentation
    """

    def __init__(self, rewrite_data_aug=False):
        self.rewrite_data_aug = rewrite_data_aug
