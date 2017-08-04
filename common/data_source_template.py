from .functor import Functor

class DataSourceTemplate(Functor):

    def __init__(self, use_augmentation=True):
        super(DataSourceTemplate, self).__init__()
        self.use_augmentation = use_augmentation
