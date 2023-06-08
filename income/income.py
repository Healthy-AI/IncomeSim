from .util import *
from .samplers import *
from .income_samplers import *
from .arm import *

def init_income_ARM(T, educ_map):
    """
    Initializes the Income ARM
    
    args: 
        T:        Transformation object. Transforms input features for fitting and sampling
        educ_map: Map between categorical education level and number of years
    """
    
    # Create ARM model
    A = MarkovARM(transformation=T)

    """ Add variables to the sampler """
    A.add_variable('time', [], 
                   ConstantSampler(value=0),
                   seq_transform_input=False,
                   seq_parents_curr=[], 
                   seq_parents_prev=['time'], 
                   seq_sampler=IncrementSampler(increment=1))
    
    A.add_variable('age', [], 
                   GaussianVariable(round_result=True, bounds=(18, np.inf)),
                   seq_transform_input=False,
                   seq_parents_curr=[], 
                   seq_parents_prev=['age'], 
                   seq_sampler=IncrementSampler(increment=1))

    A.add_variable('race', [], 
                   'multinomial', 
                   seq_parents_prev=['race'], 
                   seq_sampler=None)

    A.add_variable('sex', [], 
                   'multinomial', 
                   seq_parents_prev=['sex'], 
                   seq_sampler=None)

    A.add_variable('native-country', [], 
                   'multinomial', 
                   seq_parents_prev=['native-country'], 
                   seq_sampler=None)

    A.add_variable('education', ['age', 'race', 'sex'], 
                   LogisticSampler(multi_class='multinomial'),
                   seq_parents_prev=['education', 'training'], 
                   seq_parents_curr=['age'], 
                   seq_sampler=EducationTransition(educ_map))

    A.add_variable('education-num', ['education'], 
                   EducationNumSampler(educ_map), transform_input=False, 
                   seq_transform_input = False, 
                   seq_parents_curr=['education'], 
                   seq_sampler=EducationNumSampler(educ_map))
    
    A.add_variable('workclass', ['age', 'education', 'race', 'sex'], 
                   LogisticSampler(multi_class='multinomial'), 
                   transform_input=True, 
                   seq_transform_input = False, 
                   seq_parents_prev=['workclass'], 
                   seq_sampler=WorkclassTransition(p_stay=0.95))
    
    A.add_variable('marital-status', ['age', 'education', 'workclass', 'race'], 
                   LogisticSampler(multi_class='multinomial'), transform_input=True, seq_transform_input = False,
                   seq_parents_prev=['marital-status', 'training'], seq_parents_curr=['age'], 
                   seq_sampler=MaritalStatusTransition(a_stay=5))

    # @TODO: Make dependent on short-term study too
    occupation_sampler = LogisticSampler(multi_class='multinomial')
    A.add_variable('occupation', ['age', 'education', 'workclass', 'race', 'sex'], 
                   occupation_sampler, transform_input=True, seq_transform_input = True,
                   seq_parents_prev=['occupation','training'], seq_parents_curr=['age', 'education', 'workclass', 'race', 'sex'], 
                   seq_sampler=OccupationTransition(occupation_sampler, 0.9))

    rel_sampler = RandomForestClassifierSampler(n_estimators=20, min_samples_leaf=40)
    A.add_variable('relationship', ['age', 'education', 'workclass', 'marital-status', 'race', 'sex'], 
                   rel_sampler, seq_parents_prev=['relationship'],
                   seq_parents_curr=['age', 'education', 'workclass', 'marital-status', 'race', 'sex'],
                   seq_sampler=RelationshipTransition(rel_sampler, 0.9))
    
    hours_sampler = GaussianRegressionSampler(RandomForestRegressor(min_samples_leaf=50), 
                                              round_result=True, bounds=(0,100), std_mod=2.5)
    A.add_variable('hours-per-week', ['age', 'education', 'workclass', 'occupation', 'marital-status', 'race', 'relationship', 'sex'], 
                   hours_sampler, transform_input=True, seq_transform_input=False, 
                   seq_parents_prev=['hours-per-week'], 
                   seq_parents_curr=['age', 'education', 'workclass', 'occupation', 'marital-status', 'race', 'relationship', 'sex'], 
                   seq_sampler=HoursPerWeekTransition(T, prev_weight=0.9, sampler=hours_sampler))

    capital_sampler = ZeroOrGaussianRegressionSampler(LogisticRegression(), Ridge(), round_result=True)
    A.add_variable('capital-net', ['age', 'education', 'workclass', 'occupation', 'marital-status', 'race', 'relationship', 'sex'], 
                   capital_sampler, 
                   seq_parents_curr=['age', 'education', 'workclass', 'occupation', 'marital-status', 'race', 'relationship', 'sex'],
                   seq_parents_prev=['capital-net'],
                   seq_sampler=CapitalTransition(capital_sampler))

    TS = TrainingSampler()
    A.add_variable('training', ['age', 'sex', 'education', 'education-num', 'relationship'], TS,
                   transform_input=False, seq_transform_input=False, 
                   seq_sampler=TrainingTransition(TS), 
                   seq_parents_curr=['age', 'sex', 'education', 'education-num', 'relationship','time'],
                   seq_parents_prev=['training','income'])

    c_income_feat = ['hours-per-week', 'age', 'education', 'workclass', 'marital-status', 'occupation', 'race', 'sex', 'capital-net', 'training']
    
    # @TODO: Make dependent on short-term study too
    income_sampler = IncomeSampler()
    A.add_variable('income', c_income_feat, income_sampler, 
                   seq_sampler=IncomeTransition(income_sampler, prev_weight=0.9), 
                   seq_parents_curr=c_income_feat, 
                   seq_parents_prev=['income', 'training'],
                   transform_input=True, seq_transform_input=True, seq_fit=False)

    return A