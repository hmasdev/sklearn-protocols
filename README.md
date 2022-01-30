# Sklearn-Protocols: protocols for sklearn regressors, classifiers and transformers

Scikit-learn is one of the most commonly used machine-learning tools in python.
However, scikit-learn has a difficulty in type hints.

For example, when you write

```python
import numpy as np
from sklearn.base import BaseEstimator

def train(model: BaseEstimator, X: np.ndarray, y:np.ndarray) -> BaseEstimator:
    model.fit()
    return model

def train_(model: RegressorMixin, X: np.ndarray, y:np.ndarray) -> BaseEstimator:
    model.fit()
    return model
```

in order to train various kinds of models in a regression task, `model.fit` is not appropriately inferred with python type-checking system in Visual Studio Code.
This is because `fit` method is not included in `BaseEstimator` nor `RegressorMixin`.
It is also a reason that there are not protocols or interfaces in scikit-learn.

So, in order to solve this problem, `sklearn_protocols` provides sklearn-compatible protocols for regressors, classifiers, transformers and so on.

NOTE: [This thread](https://stackoverflow.com/a/60542986/165678329) will help you understand `sklearn_protocols`.

## Installation

### Requirements

- python >= 3.8
- scikit-learn

### User Instllation

```bash
pip install git+https://github.com/hmasdev/sklearn-protocols.git@master
```

### Basic Protocols

`sklearn_protocols` has five protocols as the basic protocols:

1. `RegressorProtocol`;
2. `ClassifierProtocol`;
3. `ClassifierWithPredictProbaProtocol`;
4. `TransformerProtocol`;
5. `ClusterProtocol`.

The definitions of those protocols are follows:

1. `RegressorProtocol` is the interface which prediction models should satisfy in regression tasks. This protocol has `get_params`, `set_params`, `fit`, `predict` and `score` as its methods;

2. `ClassifierProtocol` is the interface which prediction models should satisfy in classification tasks. This protocol has `get_params`, `set_params`, `fit`, `predict` and `score` as its methods;

3. `ClassifierWithPredictProbaProtocol` is the interface which prediction models should satisfy in classification tasks. However, `ClassifierWithPredictProbaProtocol` is different from `ClassifierProtocol`. The former has `predict_proba` method but the latter does not. That is, this protocol has `get_params`, `set_params`, `fit`, `predict` `score` and `predict_proba` as its methods;

4. `TransformerProtocol` is the interface which preprocessors should satisfy. This protocol has `get_params`, `set_params`, `fit`, `transform` and `fit_transform`;

5. `ClusterProtocol` is the interface which clustering models should satisfy in clustering tasks. This protocol has `get_params`, `set_params`, `fit` and `fit_predict`.

### How to Use

`sklearn_protocols` provides only protocols.
So, you can use the protocols as follows.

1. Type Hint:

   For example, in regression tasks, you can use RegressorProtocol like

   ```python
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn_protocols import RegressorProtocol

   def train(
       model: RegressorProtocol,
       X: np.ndarray,
       y: np.ndarray,
   ) -> RegressorProtocol:
       model.fit(Xtr, ytr)  # RegressorProtocol has `fit`.
       return model
   ```

2. Type Check:

   You can use `isinstance` or `issubclass` with the protocols to find out whether an object has methods required in machine learning tasks.
   Also, the protocols are mypy-compatible.

   ```python
   my_model = get_model()

   if isinstance(my_model, RegressorProtocol):
       print("my_models is a regressor")
   ```

If the protocols, `RegressorProtocol`, `ClassifierProtocol`, `ClassifierWithPredictProbaProtocol`, `TransformerProtocol` and `ClusterProtocol`, are insufficient for your requirements, you can create a custom protocol with `sklearn_protocols.base` and `sklearn_protocols.support`:

- `sklearn_protocols.base`;
  - `BaseEstimatorProtocol`: compatible with `sklearn.base.BaseEstimator`;
  - `BiclusterMixinProtocol`: compatible with `sklearn.base.BiclusterMixin`;
  - `ClassifierMixinProtocol`: compatible with `sklearn.base.ClassifierMixin`;
  - `ClusterMixinProtocol`: compatible with `sklearn.base.ClusterMixin`;
  - `DensityMixinProtocol`: compatible with `sklearn.base.DensityMixin`;
  - `RegressorMixinProtocol`: compatible with `sklearn.base.RegressorMixin`;
  - `TransformerMixinProtocol`: compatible with `sklearn.base.TransformerMixin`;
  - `SelectorMixinProtocol`: compatible with `sklearn.feature_selection.SelectorMixin`;
- `sklearn_protocols.support`;
  - `SupportFit`: : only supporting `fit` method;
  - `SupportPredict`: only supporting `predict` method;
  - `SupportPredictProba`: only supporting `predict_proba` method;
  - `SupportPredictLogProba`: only supporting `predict_log_proba` method;
  - `SupportTransform`: only supporting `transform` method;
  - `SupportFitPredict`: only supporting `fit_predict` method;
  - `SupportFitTransform`: only supporting `fit_transform` method.

## Contribution Guide

### Requirements

- python >= 3.8
- pipenv

### Setup

After `fork`,

```bash
$ git clone https://github.com/{YOUR_ACCOUNT}/sklearn-protocols.git
$ cd sklearn-protocols
$ pipenv install --dev
```

### Issues

- For any bugs, use [BUG REPORT](https://github.com/hmasdev/sklearn-protocols/issues/new?assignees=&labels=bug&template=bug_report.md&title=%5BBUG%5D) to create an issue.

- For any enhancement, use [FEATURE REQUEST](https://github.com/hmasdev/sklearn-protocols/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=) to create an issue.

- For other topics, create an issue with a clear and concise description.

### Pull Request

1. Fork (https://github.com/hmasdev/sklearn-protocols/fork);
2. Create your feature branch (git checkout -b feautre/xxxx);
3. Test codes according to Test Subsection;
4. Commit your changes (git commit -am 'Add xxxx feature);
5. Push to the branch (git push origin feature/xxxx);
6. Create new Pull Request

### Test

```bash
$ pipenv run flake8
$ pipenv run mypy .
$ pipenv run pytest
```

## LICENSE

MIT

## Authors

[hmasdev](https://github.com/hmasdev)
