import numpy as np
import math

from .lowessi import Lowessi


class OverfittingDetector(object):
    """
    Overfitting detector

    Parameters:
    ----------
    epoch_tail : int or None
        number of epochs for analysing of convergence
    min_num_epoch : int or None
        pure minimum number of epochs that should be without extra logic
    bigger : bool
        flag for minization or maximization
    lowess_factor : float
        factor for LOWESS smoothing of tail
    tol : float
        TOL for overfitting detection
    """

    def __init__(self,
                 epoch_tail=None,
                 min_num_epoch=0,
                 bigger=True,
                 lowess_factor=0.5,
                 tol=1e-5):
        assert lowess_factor > 0.0 and lowess_factor < 1.0
        assert tol >= 0
        assert epoch_tail is None or epoch_tail > 8
        self.epoch_tail = epoch_tail
        self.min_num_epoch = min_num_epoch
        self.bigger = bigger
        self.tol = tol
        self.invalid_params = self.epoch_tail is None

        if self.invalid_params:
            return

        self.y = []
        self.better = []
        self.y_tail = []
        self.y_smooth = []

        margin = int(math.ceil(0.05 * epoch_tail))
        self.epoch_tail += 2 * margin
        self.margin1 = margin
        self.margin2 = self.epoch_tail - margin

        self.x = np.linspace(0.0, 1.0, self.epoch_tail)
        self.xx, self.w = Lowessi.preprocess(self.x, f=lowess_factor)

    def check(self, value, is_better):
        """
        Check for the next value.

        Parameters:
        ----------
        value : float
            checked value
        is_better : bool
            is this value better than the previous

        Returns:
        ----------
        is_detected : bool
            must we do break due to overfitting
        """

        if self.invalid_params:
            return False
        self.y.append(value)
        self.better.append(is_better)
        if len(self.y) < self.min_num_epoch:
            return False
        if len(self.y) < self.epoch_tail:
            return False
        if np.any(self.better[-self.epoch_tail:]):
            return False

        self.y_tail = np.array(self.y[-self.epoch_tail:])
        self.y_smooth = Lowessi.update(self.y_tail, self.x, self.xx, self.w)
        y_diff = np.diff(self.y_smooth[self.margin1:self.margin2])
        y_diff_mask = (y_diff > self.tol) if self.bigger else (
            y_diff < -self.tol)
        return not np.any(y_diff_mask)
