import inspect 
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


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

    elif e in ['Ridge', 'ridge']:
        return Ridge()

    elif e in ['RandomForestRegressor', 'rfr']:
        return RandomForestRegressor()

    elif e in ['T-learner', 'T_learner', 't_learner']:
        return T_learner()

    else: 
        raise Exception('Unknown estimator %s' % e)


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

    #def score. Left out.   

    def set_params(self, **params):
        """ Set the parameter of the estimator and base estimators """

        if 'estimator0' in params: 
            self.estimator0 = get_estimator(params['estimator0'])
            del params['estimator0']

        if 'estimator1' in params: 
            self.estimator1 = get_estimator(params['estimator1'])
            del params['estimator1']

        super().set_params(**params)
    

class T_learner(PotentialOutcomeEstimator):
    """ Implements the T-learner meta learner """ 

    def __init__(self, estimator0='ridge', estimator1='ridge', c_int='intervention', c_out='outcome', c_adj=[], v_int0=0, v_int1=1):

        self.estimator0 = get_estimator(estimator0)
        self.estimator1 = get_estimator(estimator1)
        self.c_int = c_int
        self.c_out = c_out
        self.c_adj = c_adj
        self.v_int0 = v_int0
        self.v_int1 = v_int1
        
    def fit(self, x, y):
        
        # @TODO: Could do the dicotomization here instead?
        # @TODO: For splitting keys if needed: key, delim, sub_key = key.partition("__")

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
    
        I0 = x['%s__%s' % (self.c_int, str(self.v_int0))]==1
        I1 = x['%s__%s' % (self.c_int, str(self.v_int1))]==1

        if (I0.sum() == 0) or (I1.sum() ==0):
            raise Exception('One of the intervention options has 0 samples')

        self.estimator0.fit(x[I0][c_adjs], y[I0])
        self.estimator1.fit(x[I1][c_adjs], y[I1])

        return self

    def predict(self, x):

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj]
    
        I0 = x['%s__%s' % (self.c_int, str(self.v_int0))]==1
        I1 = x['%s__%s' % (self.c_int, str(self.v_int1))]==1

        yp = np.zeros(x.shape[0])
        if I0.sum()>0:
            yp[I0] = self.estimator0.predict(x[I0][c_adjs])
        if I1.sum()>0:
            yp[I1] = self.estimator1.predict(x[I1][c_adjs])

        return yp



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