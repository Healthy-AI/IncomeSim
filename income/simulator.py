from .util import *
from .samplers import *
from .income_samplers import *
from .arm import *
from . import income_samplers


def instantiate_sampler(cfg, base_sampler=None, transformation=None):
    
    if cfg == 'gaussian':
        return GaussianVariable()
    if cfg == 'multinomial':
        return MultinomialVariable()
    if cfg is None or cfg == 'None': 
        return None
    
    cname = cfg.type
    args = {k:v for k,v in cfg.__dict__.items() if not k in ['type']}
    for k,v in args.items():
        if v == '#init_sampler':
            args[k] = base_sampler
        if v == '#transformation':
            args[k] = transformation
            
    clf = getattr(income_samplers, cname)(**args)
    
    return clf

def create_simulator(cfg, transformation):
    """
    Initializes a simulator based on a configuration file
    
    args: 
        cfg:            Configuration (nested namespace)
        transformation: Transformation object. Transforms input features for fitting and sampling
    """
    
    # Create ARM model
    A = MarkovARM(transformation=transformation)

    # Add variables, one at a time
    variables = cfg.variables.__dict__
    for k, v in variables.items():
        args = {ak:av for ak,av in v.__dict__.items() if not ak in ['sampler', 'seq_sampler']}
        
        sampler = instantiate_sampler(v.sampler)
        seq_sampler  = instantiate_sampler(v.seq_sampler, base_sampler=sampler, transformation=transformation)
        
        A.add_variable(name=k, sampler=sampler, seq_sampler=seq_sampler, **args)

    return A