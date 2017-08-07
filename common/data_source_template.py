from .functor import Functor


class DataSourceTemplate(Functor):
    """
    Base class for DataSource (class, which provides training/validation data iterators).

    Parameters:
    ----------
    use_augmentation : bool
        do use augmentation during training (for training dataset)
    """

    def __init__(self, use_augmentation=True):
        super(DataSourceTemplate, self).__init__()
        self.use_augmentation = use_augmentation
