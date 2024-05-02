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

def sample(cfg):

    # Load data
    D_tr, c_cat, c_num, c_out, c_features = load_income_data(cfg.data.path)
    
    # Load simulator
    print('Loading model ...')
    A = load_model(cfg.simulator.path, cfg.simulator.label)
    
    # Set propensity model
    if cfg.samples.policy in ['no', 'full']:
        if cfg.samples.policy == 'no':
            policy = 'No studies'
        elif cfg.samples.policy == 'full':
            policy = 'Full-time studies'
        A.replace_variable('studies', [], ConstantSampler(policy), 
                       seq_sampler=StudiesTransition(StudiesSampler()), 
                       seq_parents_curr=['age', 'sex', 'education', 'education-num', 'relationship', 'time'], 
                       seq_parents_prev=['studies','income'], 
                       seq_transform_input=False)
    elif cfg.samples.policy == 'default':
        pass
    else:
        raise Exception('Unknown sampling policy \'%s\'. Aborting.' % cfg.samples.policy)

    # Sample observations
    np.random.seed(cfg.samples.seed)
    print('Sampling observations ...')
    S = A.sample(cfg.samples.n_samples, T=cfg.samples.horizon)
   
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


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Sample from IncomeSCM simulator')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='config_v1.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    sample(cfg)
    

    