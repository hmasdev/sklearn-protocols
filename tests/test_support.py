from sklearn_protocols.support import (
    SupportFit,
    SupportPredict,
    SupportPredictProba,
    SupportPredictLogProba,
    SupportTransform,
    SupportFitPredict,
    SupportFitTransform,
)


def test_support_fit():
    hasattr(SupportFit, "fit")


def test_support_predict():
    hasattr(SupportPredict, "predict")


def test_support_predict_proba():
    hasattr(SupportPredictProba, "predict_proba")


def test_support_predict_log_proba():
    hasattr(SupportPredictLogProba, "predict_log_proba")


def test_support_transform():
    hasattr(SupportTransform, "transform")


def test_support_fit_predict():
    hasattr(SupportFitPredict, "fit_predict")


def test_support_fit_transform():
    hasattr(SupportFitTransform, "fit_transform")
