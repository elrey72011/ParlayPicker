"""Probability calibration utilities."""

from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, preds, outcomes):
        self.calibrator.fit(preds, outcomes)

    def transform(self, preds):
        return self.calibrator.transform(preds)
