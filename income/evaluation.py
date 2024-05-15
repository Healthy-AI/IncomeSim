import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error

def cate_evaluation(clf, df0, df1, c_cov, c_int, c_out, n_bootstrap=1000, alpha=0.05):

    n = df0.shape[0]

    # Check that df0 and df1 are aligned by sampling random rows and check that they are the same
    for i in np.random.choice(n, 10):
        if np.any(df0.iloc[i][c_cov] != df1.iloc[i][c_cov]):
            print(df0.iloc[i][c_cov] != df1.iloc[i][c_cov])
            raise Exception('Dataframes for control and treated counterfactuals not matched')
    

    dft0 = clf[:-1].transform(df0[c_cov+[c_int]])
    dft1 = clf[:-1].transform(df1[c_cov+[c_int]])
    y0 = clf[-1].predict_outcomes(dft0)
    y1 = clf[-1].predict_outcomes(dft1)

    cate_est = y1-y0
    cate_sample = df1[c_out]-df0[c_out]

    ate_est = cate_est.mean()
    ate_sample = cate_sample.mean()

    scoring = {"R2": r2_score, 
               "RMSE": root_mean_squared_error, 
               "MSE": mean_squared_error}

    df = pd.DataFrame({'cate_est': cate_est, 'cate_sample': cate_sample})
    
    rows = []
    for i in range(n_bootstrap):
        dfr = resample(df, n_samples=df.shape[0])

        ate_est_r    = dfr['cate_est'].mean()
        ate_sample_r = dfr['cate_sample'].mean()

        # Cate measures
        row = { ('CATE_%s_r' % k):s(dfr['cate_sample'], dfr['cate_est']) for k, s in scoring.items() }
        row['ATE_AE_r'] = np.abs(ate_est_r - ate_sample_r)
        row['ATE_SE_r'] = np.abs(ate_est_r - ate_sample_r)**2

        rows.append(row)

    Ra = pd.DataFrame(rows)
    R = Ra.mean()
    for c in Ra.columns: 
        R[c+'_l'] = np.percentile(Ra[c], alpha/2*100)
        R[c+'_u'] = np.percentile(Ra[c], (1-alpha/2) * 100)
        
    for k, s in scoring.items():
        R['CATE_'+k] = s(cate_sample, cate_est)
        
    R['ATE_AE'] = np.abs(ate_est-ate_sample)
    R['ATE_SE'] = np.abs(ate_est-ate_sample)**2
    R['ATE_True'] = ate_sample

    R = R.to_frame().T
    R = R[np.sort(R.columns)]

    return R