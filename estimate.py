import argparse
import sklearn
import pandas as pd
import numpy as np
sklearn.set_config(transform_output="pandas")
from sklearn.exceptions import ConvergenceWarning

# Move to estimators
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


from income.data import *
from income.util import *
from income.samplers import *
from income.income_samplers import *
from income.arm import *
from income.income import *
from income.estimators import *

def run_experiment(cfg):
    """ Estimate the causal effect of interventions and evaluate the results
    """

    obs_path = os.path.join(cfg.data.path, cfg.data.observational)
    df_obs = pd.read_pickle(obs_path)

    c_cov = cfg.experiment.covariates
    c_tre = cfg.experiment.treatment
    c_out = cfg.experiment.outcome

    # Fetch numeric features
    numeric_features = [k for k in c_cov if df_obs[k].dtype != 'category']
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # Fetch categorical and binary features
    categorical_features = [k for k in c_cov if df_obs[k].dtype == 'category']
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary', 
                                      feature_name_combiner=name_combiner)),
        ]
    )

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ], verbose_feature_names_out=False
    )

    pipe = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
    pipe.fit(df_obs[c_cov], df_obs[c_out])


    """

    # Fit and apply preprocessing (@note: This is on the entire data set. No other indication in JAMA paper)
    df_pre = preprocessor.fit_transform(df)
    for c in categorical_features: 
        if c+'__nan' in df_pre.columns:
            cs = [k for k in df_pre.columns if k.rsplit('__',maxsplit=1)[0]==c and not k.endswith('__nan')]
            nans = df_pre[c+'__nan']
            df_pre.loc[nans==1,cs]=np.nan
    
            df_pre = df_pre.drop(columns=[c+'__nan'])

    # Construct map of categorical features
    cout = df_pre.columns
    cat_map = {}
    for c in cout:
        s = c.rsplit('__', maxsplit=1)[0]
        if s in categorical_features:
            cat_map[s] = cat_map.get(s, []) + [c]

    return df_pre, preprocessor, cat_map

    """


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Estimate causal effects from IncomeSim samples')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='estimation.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    run_experiment(cfg)
    

    