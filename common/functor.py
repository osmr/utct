class Functor(object):

    def __init__(self):
        self.param_bounds = {}
        self.params = {}

    def update_param_bounds(self, new_param_bounds):
        self.param_bounds = dict((k, new_param_bounds[k] if k in new_param_bounds else v) for k, v in self.param_bounds.items())

    def update_params(self, new_params):
        self.params = dict((k, new_params[k] if k in new_params else v) for k, v in self.params.items())

    def __call__(self, **kwargs):
        raise NotImplementedError()
