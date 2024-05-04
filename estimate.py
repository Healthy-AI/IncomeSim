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

def estimate(cfg):
    """ Estimate the causal effect of interventions
    """
    

    


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Estimate causal effects from IncomeSim samples')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='estimation.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    sample(cfg)
    

    