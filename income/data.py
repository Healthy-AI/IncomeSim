import os, sys
import pandas as pd
import numpy as np

from .income_samplers import *

def load_income_data(folder='data/adult/', data_only=False):
    """
    Loads the "Adult" income dataset and returns

    args
        folder: The folder where the data files are
        data_only: If true, returns only the dataframe for the training portion

    returns
        D_tr: Training dataframe
        c_cat: Categorical columns
        c_num: Numerical columns
        c_out: Outcome column
        c_features: Feature columns
    """

    train_file = os.path.join(folder, 'adult.data')
    col_file = os.path.join(folder, 'adult.names')
    test_file = os.path.join(folder, 'adult.test')

    # Load data
    D_tr = pd.read_csv(train_file, header=None, delimiter=', ', engine='python')

    # Column names
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income>50k']
    c_cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    #c_num = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    c_bin = ['income>50k']
    c_out = 'income>50k'

    D_tr.columns = columns

    # Remove redundant features
    D_tr['capital-net'] = D_tr['capital-gain'] - D_tr['capital-loss']
    D_tr = D_tr.drop(columns=['fnlwgt', 'capital-gain', 'capital-loss'])
    c_num = ['age', 'education-num', 'capital-net', 'hours-per-week']

    D_tr = D_tr.replace('<=50K', 0).replace('>50K', 1).replace('?', np.nan).replace(' ', '')
    D_tr = D_tr.replace('Married-civ-spouse', 'Married').replace('Married-AF-spouse', 'Married').replace('Married-spouse-absent', 'Married')

    # Drop nan values
    D_tr = D_tr.dropna()

    # Add index columns
    D_tr['id'] = range(D_tr.shape[0])
    D_tr['time'] = 0

    # Feature columns
    c_features = [c for c in list(D_tr.columns) if c not in ['income>50k', c_out]]

    # Construct map between education-num and education
    # educ_map = dict(D_tr[['education', 'education-num']].groupby('education', as_index=False).mean().sort_values('education-num').values)
    # @TODO: No longer used. Hard-coded in income_samplers.py instead

    c_cat += ['studies']
    L_studies = StudiesSampler.classes_
    D_tr['studies'] = (L_studies*int(np.ceil(D_tr.shape[0]/len(L_studies))))[:D_tr.shape[0]]

    D_tr['income'] = D_tr['income>50k'].copy()

    # Set data type to category for categorical features
    D_tr[c_cat] = D_tr[c_cat].astype('category')

    if data_only:
        return D_tr
    else:
        return D_tr, c_cat, c_num, c_out, c_features
