import inspect 
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import xgboost as xgb


def name_combiner(x,y):
    """ Used by column transformer for one-hot encoder """
    return str(x)+'__'+str(y)

def get_estimator(e):
    """ Returns an estimator given the provided string """ 

    if inspect.isclass(e) and issubclass(e, BaseEstimator):
        return e()

    if isinstance(e, BaseEstimator):
        return e

    if type(e) != str:
        raise Exception('Unknown estimator specification: %s' % str(e)) 
        #@TODO: Should support a class and instance of estimator

    if e in ['LinearRegression', 'ols']:
        return LinearRegression()

    if e in ['LogisticRegression', 'lr']:
        return LogisticRegression()

    elif e in ['Ridge', 'ridge']:
        return Ridge()

    elif e in ['RandomForestClassifier', 'rfc']:
        return RandomForestClassifier()

    elif e in ['RandomForestRegressor', 'rfr']:
        return RandomForestRegressor()

    elif e in ['XGBRegressor', 'xgbr']:
        return xgb.XGBRegressor()

    elif e in ['T-learner', 'T_learner', 't_learner']:
        return T_learner()

    elif e in ['S-learner', 'S_learner', 's_learner']:
        return S_learner()

    elif e in ['IPWEstimator', 'ipw', 'IPW']:
        return IPWEstimator()

    else: 
        raise Exception('Unknown estimator %s' % e)

class IPWEstimator(BaseEstimator, ClassifierMixin):
    """ Implements the inverse propensity weighting (IPW) estimator of average outcome """ 

    def __init__(self, base_estimator='lr', c_int='intervention', c_out='outcome', c_adj=[], weighted=True, v_int0=0, v_int1=1):
        
        self.base_estimator = get_estimator(base_estimator)
        self.c_int = c_int
        self.c_out = c_out
        self.c_adj = c_adj
        self.v_int0 = v_int0
        self.v_int1 = v_int1
        self.weighted = weighted
        
    def _estimate(self, X, t, y):
        """ Estimates mean potential outcomes """ 
        
        e = self.base_estimator.predict_proba(X)[:,1]

        t = 1*(t==self.v_int1)
        n = X.shape[0]
        n0 = (1-t).sum()
        n1 = n-n0
        
        w0 = ((1-t)/(1-e))/n
        w1 = (t/e)/n

        if self.weighted:
            w0 = w0/(w0.sum())
            w1 = w1/(w1.sum())
        
        self.y0 = np.sum(w0*y)
        self.y1 = np.sum(w1*y)
        self.ate = self.y1 - self.y0

        self.w = (t*w1 + (1-t)*w0)*n
        
        
    def fit(self, x, t, sample_weight=None):
        """ Fits a propensity model and estimates mean potential outcomes """ 

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
        
        X = x[c_adjs]
        y = x[self.c_out]
        
        self.base_estimator.fit(X, t)
        self.classes_ = self.base_estimator.classes_
        
        self._estimate(X, t, y)

        return self
        
        
    def predict(self, x):
        """ Predicts the treatment using the propensity model """ 
        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
        X = x[c_adjs]
        
        return self.base_estimator.predict(X)

    def predict_proba(self, x):
        """ Predicts the treatment probability using the propensity model """ 
        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
        X = x[c_adjs]
        
        return self.base_estimator.predict_proba(X)

    def predict_propensity(self, x):
        """ Returns the estimated probability of the treated intervention """

        return self.predict_proba(x)[:,1]

    def predict_outcomes(self, x):
        """ Predicts the potential outcomes given covariates and treatments in x """
        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
        
        X = x[c_adjs]
        t = x[self.c_int + '__%s' % self.v_int1]
        
        y = t*self.y1 + (1-t)*self.y0

        return y

    def set_params(self, **params):
        """ Set the parameter of the estimator and base estimators """

        # To enable setting base estimator parameter by string
        if 'base_estimator' in params: 
            self.base_estimator = get_estimator(params['base_estimator'])
            del params['base_estimator']

        super().set_params(**params)
        return self


class PotentialOutcomeEstimator(BaseEstimator):
    """ Interface for estimators of potential outcomes of interventions """

    def __init__(self):
        pass

    def fit(self, x, y, sample_weight=None):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass

    def predict_outcomes(self, x):
        return self.predict(x)

    #def score. Left out.   

    def set_params(self, **params):
        """ Set the parameter of the estimator and base estimators """

        super().set_params(**params)
        return self
    
class S_learner(PotentialOutcomeEstimator):
    """ Implements the S-learner meta learner """ 

    def __init__(self, base_estimator='ridge', c_int='intervention', c_out='outcome', c_adj=[], v_int0=0, v_int1=1):

        self.base_estimator = get_estimator(base_estimator)
        self.c_int = c_int
        self.c_out = c_out
        self.c_adj = c_adj
        self.v_int0 = v_int0
        self.v_int1 = v_int1
        
    def fit(self, x, y):
        """ Fits the base estimator """
        
        adj = self.c_adj + [self.c_int] # Add intervention to adjustment columns
        c_adjs = [c for c in x.columns if c in adj or c.partition('__')[0] in adj] # @TODO: This won't warn if there are columns in adj not represented in columns

        self.base_estimator.fit(x[c_adjs], y)

        return self

    def predict(self, x):
        """ Predicts the potential outcomes given covariates and treatments in x """

        adj = self.c_adj + [self.c_int] # Add intervention to adjustment columns
        c_adjs = [c for c in x.columns if c in adj or c.partition('__')[0] in adj]

        yp = self.base_estimator.predict(x[c_adjs])

        return yp

    def set_params(self, **params):
        """ Set the parameter of the estimator and base estimators """

        # To enable setting base estimator parameter by string
        if 'base_estimator' in params: 
            self.base_estimator = get_estimator(params['base_estimator'])
            del params['base_estimator']

        super().set_params(**params)
        return self



class T_learner(PotentialOutcomeEstimator):
    """ Implements the T-learner meta learner """ 

    def __init__(self, base_estimator0='ridge', base_estimator1='ridge', c_int='intervention', c_out='outcome', c_adj=[], v_int0=0, v_int1=1):

        self.base_estimator0 = get_estimator(base_estimator0)
        self.base_estimator1 = get_estimator(base_estimator1)
        self.c_int = c_int
        self.c_out = c_out
        self.c_adj = c_adj
        self.v_int0 = v_int0
        self.v_int1 = v_int1
        
    def fit(self, x, y):
        """ Fits the base estimators """
        
        # @TODO: Could do the dicotomization here instead?
        # @TODO: For splitting keys if needed: key, delim, sub_key = key.partition("__")

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
    
        I0 = x['%s__%s' % (self.c_int, str(self.v_int0))]==1
        I1 = x['%s__%s' % (self.c_int, str(self.v_int1))]==1

        if (I0.sum() == 0) or (I1.sum() ==0):
            raise Exception('One of the intervention options has 0 samples')

        self.base_estimator0.fit(x[I0][c_adjs], y[I0])
        self.base_estimator1.fit(x[I1][c_adjs], y[I1])

        return self

    def predict(self, x):
        """ Predicts the potential outcomes given covariates and treatments in x """

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
    
        I0 = x['%s__%s' % (self.c_int, str(self.v_int0))]==1
        I1 = x['%s__%s' % (self.c_int, str(self.v_int1))]==1

        yp = np.zeros(x.shape[0])
        if I0.sum()>0:
            yp[I0] = self.base_estimator0.predict(x[I0][c_adjs])
        if I1.sum()>0:
            yp[I1] = self.base_estimator1.predict(x[I1][c_adjs])

        return yp

    def set_params(self, **params):
        """ Set the parameter of the estimator and base estimators """

        # To enable setting base estimator parameter by string
        if 'base_estimator0' in params: 
            self.base_estimator0 = get_estimator(params['base_estimator0'])
            del params['base_estimator0']

        if 'base_estimator1' in params: 
            self.base_estimator1 = get_estimator(params['base_estimator1'])
            del params['base_estimator1']

        super().set_params(**params)
        return self



def get_pipeline(estimator, c_num, c_cat):
    """ Creates training pipeline for estimator given column description """

    # Get estimator
    if not isinstance(estimator, BaseEstimator):
        if type(estimator) == str:
            estimator = get_estimator(estimator)
        else:
            raise Exception('Unknown estimator specification: %s' % str(estimator))
    
    # Create transformers
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )   
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", 
                            sparse_output=False, feature_name_combiner=name_combiner)),
        ]
    )

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, c_num),
            ("cat", categorical_transformer, c_cat),
        ], verbose_feature_names_out=False
    )

    pipe = Pipeline([('preprocessor', preprocessor), ('estimator', estimator)])

    return pipe



""" @TODO: Useful??
# Construct map of categorical features
    cout = df_pre.columns
    cat_map = {}
    for c in cout:
        s = c.rsplit('__', maxsplit=1)[0]
        if s in categorical_features:
            cat_map[s] = cat_map.get(s, []) + [c]

    return df_pre, preprocessor, cat_map
"""