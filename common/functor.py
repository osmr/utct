class Functor(object):
    """
    A base class that provides the ability to optimize parameters in its descendants. It also implements functor functionality.
    """

    def __init__(self):
        self.param_bounds = {}
        self.params = {}

    def update_param_bounds(self, new_param_bounds):
        """
        Updating parameter limits (actually combining them with existing values).

        Parameters:
        ----------
        new_param_bounds : dict
            new values of parameter limits
        """
        self.param_bounds = dict((k, new_param_bounds[k] if k in new_param_bounds else v) for k, v in self.param_bounds.items())

    def update_params(self, new_params):
        """
        Updating parameters (actually combining them with existing values).

        Parameters:
        ----------
        new_params : dict
            new values of parameters
        """
        self.params = dict((k, new_params[k] if k in new_params else v) for k, v in self.params.items())

    def __call__(self, **kwargs):
        """
        Abstract implementation of functor functionality.
        """
        raise NotImplementedError()
