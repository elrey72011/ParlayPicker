from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, preds, outcomes):
        self.model.fit(preds, outcomes)

    def transform(self, preds):
        return self.model.transform(preds)
