import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error

def cate_evaluation(clf, df0, df1, c_cov, c_int, c_out):

    n = df0.shape[0]

    # Check that df0 and df1 are aligned by sampling random rows and check that they are the same
    for i in np.random.choice(n, 10):
        if np.any(df0.iloc[i][c_cov] != df1.iloc[i][c_cov]):
            print(df0.iloc[i][c_cov] != df1.iloc[i][c_cov])
            raise Exception('Dataframes for control and treated counterfactuals not matched')
    
    y0 = clf.predict(df0[c_cov+[c_int]])
    y1 = clf.predict(df1[c_cov+[c_int]])

    cate_est = y1-y0
    cate_sample = df1[c_out]-df0[c_out]

    ate_est = cate_est.mean()
    ate_sample = cate_sample.mean()

    scoring = {"R2": r2_score, 
               "RMSE": root_mean_squared_error, 
               "MSE": mean_squared_error}

    R = { (k+'_CATE'):s(cate_sample, cate_est) for k, s in scoring.items() }
    R['AE_ATE'] = np.abs(ate_est-ate_sample)
    R['SE_ATE'] = np.abs(ate_est-ate_sample)**2
    R['True_ATE'] = ate_sample

    df = pd.DataFrame({k:[v] for k,v in R.items()})
    print(df)

    return df