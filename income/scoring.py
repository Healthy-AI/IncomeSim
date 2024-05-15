from sklearn.metrics._scorer import _Scorer, _BaseScorer
from sklearn.base import is_regressor
from sklearn.utils.validation import _check_response_method

class _BaseScorerDF(_BaseScorer):
    def __init__(self, score_func, sign, kwargs, response_method="predict", c_target="outcome"):
        super().__init__(score_func, sign, kwargs, response_method)
        self._c_target = c_target

class _DFScorer(_BaseScorerDF):
    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        """Evaluate the response method of `estimator` on `X` and `y_true`.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y_true : None
            Ignored

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        pos_label = None if is_regressor(estimator) else self._get_pos_label()
        response_method = _check_response_method(estimator, self._response_method)
        y_pred = method_caller(
            estimator, response_method.__name__, X, pos_label=pos_label
        )

        y_true = X[self._c_target]

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)

def make_scorer_df(
    score_func,
    *,
    response_method=None,
    greater_is_better=True,
    c_target="outcome",
    **kwargs,
):
    sign = 1 if greater_is_better else -1
    return _DFScorer(score_func, sign, kwargs, response_method, c_target)