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
    parser.add_argument('-p', '--policy', type=str, default='default', dest='policy', help='Starting policy')

    args = parser.parse_args()
    
    # Load data
    D_tr, c_cat, c_num, c_out, c_features = load_income_data(args.datadir)
    
    # Transformation
    T = SubsetTransformer()
    D_prev = D_tr.copy().rename(columns=dict([(c,c+'#prev') for c in D_tr.columns if not c == 'id']))
    D_tran = pd.merge(D_tr, D_prev, on='id')    
    D_tran['income#prev'] = np.random.randn(D_tran.shape[0])    # Ensure that we avoid transforming income#prev
    T.fit(D_tran)
    
    # Create ARM model
    print('Initializing ARM model ...')
    A = init_income_ARM(T)

    # Fit to observed data 
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    print('Fitting ARM model (suppressing warnings) ...')
    A.fit(D_tr)
    
    # Set propensity model
    if args.policy in ['no', 'full']:
        if args.policy == 'no':
            policy = 'No studies'
        elif args.policy == 'full':
            policy = 'Full-time studies'
        A.replace_variable('studies', [], ConstantSampler(policy), 
                       seq_sampler=StudiesTransition(StudiesSampler()), 
                       seq_parents_curr=['age', 'sex', 'education', 'education-num', 'relationship', 'time'], 
                       seq_parents_prev=['studies','income'], 
                       seq_transform_input=False)

    # Sample observations
    np.random.seed(args.seed)
    print('Sampling observations ...')
    S = A.sample(args.n_samples, T=args.horizon)
   
    # Prep data
    df0 = S[S['time']==0]
    df = df0.copy().rename(columns={'income': 'income_current'})

    Tend = args.horizon-1
    df['income'] = S[S['time']==Tend]['income'].values
    
    # Make categorical columns have the right type
    c_cols = ['native-country', 'sex', 'race', 'education', 
              'studies', 'workclass', 'occupation', 'marital-status', 'relationship']
    df[c_cols] = df[c_cols].astype('category')

    # Drop index columns
    df = df.drop(columns=['time','id'])
    
    # Reorder columns
    special_cols = ['studies', 'income']
    df = df[[c for c in df.columns if c not in special_cols] + special_cols]
    
    # Save data to file
    fname = '%s_%s_n%d_T%d_s%d.pkl' % (args.label, args.policy, args.n_samples, args.horizon, args.seed)
    df.to_pickle('data/%s' % fname)
    print('Saved result to: %s' % fname)
    