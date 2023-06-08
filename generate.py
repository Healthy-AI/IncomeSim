import argparse
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.exceptions import ConvergenceWarning

from income.data import *
from income.util import *
from income.samplers import *
from income.income_samplers import *
from income.arm import *
from income.income import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate IncomeSim')
    parser.add_argument('-l', '--label', type=str, dest='label', help='Label of dataset', default='income')
    parser.add_argument('-n', '--n_samples', type=int, default=100, dest='n_samples', help='Number of instances')
    parser.add_argument('-T', '--horizon', type=int, default=1, dest='horizon', help='Length of horizon')
    parser.add_argument('-s', '--seed', type=int, default=0, dest='seed', help='Random seed')
    parser.add_argument('-d', '--datadir', type=str, default='data/income', dest='datadir', help='Data directory')

    args = parser.parse_args()
    
    # Load data
    D_tr, c_cat, c_num, c_out, c_features, educ_map = load_income_data(args.datadir)
    
    # Transformation
    T = SubsetTransformer()
    D_prev = D_tr.copy().rename(columns=dict([(c,c+'#prev') for c in D_tr.columns if not c == 'id']))
    D_tran = pd.merge(D_tr, D_prev, on='id')    
    D_tran['income#prev'] = np.random.randn(D_tran.shape[0])    # Ensure that we avoid transforming income#prev
    T.fit(D_tran)
    
    # Create ARM model
    print('Initializing ARM model ...')
    A = init_income_ARM(T, educ_map)

    # Fit to observed data 
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    print('Fitting ARM model (suppressing warnings) ...')
    A.fit(D_tr)
    
    # Sample observations
    np.random.seed(args.seed)
    print('Sampling observations ...')
    S = A.sample(args.n_samples, T=args.horizon)
    
    fname = '%s_m%d_T%d_s%d.pkl' % (args.label, args.n_samples, args.horizon, args.seed)
    S.to_pickle('data/%s' % fname)
    print('Saved result to: %s' % fname)