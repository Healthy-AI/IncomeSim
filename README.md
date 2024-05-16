# IncomeSCM

IncomeSCM is a time-series simulator based on the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult).

## Using the CATE estimation data set (IncomeSCM-1.0.CATE)

The IncomeSCM-1.0.CATE data set is sampled from the IncomeSCM-1.0 simulator, fit with the ```config_v1.yml``` configuration file.

### Data set description

The data set represents 13 variables extracted from the 1994 US Census bureau database, as well as a hand-designed "studies" variable. 

**Covariates**

| Column  | Description | Type |
| ------------- | ------------- | ------------- |
| native-country  | Native country  | Categorical |
| sex  | Sex (as reported in census) | Categorical |
| race  | Race (as reported in census)  | Categorical |
| age  | Age  | Numeric |
| education  | Education type (e.g., Bachelors)  | Categorical |
| education-num  | Education (numeric representation)  | Numeric |
| workclass  | Workclass (e.g., private, self-employed)  | Categorical |
| occupation  | Occupation (e.g., Tech-support)  | Categorical |
| marital-status  | Marital status (e.g., married)  | Categorical |
| relationship  | Relationship type (e.g., wife)  | Categorical |
| capital-net  | Net capital gains  | Numeric |
| hours-per-week  | Number of work hours per week | Numeric |
| income_prev  | Income the previous year (USD)  | Numeric |
| studies_prev  | Studies the previous year  | Categorical |

**Intervention**
| Column  | Description | Type |
| ------------- | ------------- | ------------- |
| studies  | Studies the current year (e.g., Full-time studies)  | Categorical |

**Outcome**
| Column  | Description | Type |
| ------------- | ------------- | ------------- |
| income  | Income 5 years after the intervention (USD)   | Numeric |

### Task description

The goal is to estimate the causal effect on ```income``` ($Y$) after intervening on ```studies``` with "Full-time studies" ($T=1$), relative to "No studies" ($T=0$),
$$\Delta = Y(1) - Y(0),$$
where $Y(t)$ is the potential outcome of intervening with $T\leftarrow t$. In particular, we are interested in the conditional average treatment effect (CATE),
$$\mathrm{CATE}(x) = \mathbb{E}[\Delta \mid X=x]$$
where $X$ is a given set of covariates. For this, we consider three main conditioning sets: 
1. $X$ is the set of all pre-intervention covariates
2. $X$ is the set of direct causes of $T$
3. $X$ is a subset of covariates which is an invalid adjustment set. Specifically, $X = (\mathrm{age}_1, \mathrm{education}_1, \mathrm{income}_1)$.

In addition, we seek to estimate the average treatment effect (ATE), $$\mathrm{ATE} = \mathbb{E}[\Delta]$$ using the first two conditioning sets above for adjustment. 

We measure the quality in estimates by the $R^2$, MSE, RMSE for CATE and the absolute error (AE) for ATE. 

### File description

 The main data set files are:
  * ```IncomeSCM-1.0.CATE_default.pkl``` (V1 simulator, default policy, 50 000 samples, horizon T=5, seed=0)
  * ```IncomeSCM-1.0.CATE_no.pkl``` (V1 simulator, "Full" policy, 50 000 samples, horizon T=5, seed=1)
  * ```IncomeSCM-1.0.CATE_full.pkl``` (V1 simulator, "No" policy, 50 000 samples, horizon T=5, seed=1)
  * All three files are contained in [IncomeSCM-1.0.CATE.zip](samples/IncomeSCM-1.0.CATE.zip)
* The "default" policy data set represents observational data for causal effect estimators to learn from.
* The "full" and "no" policy data sets represent samples observed under alternative interventions (T<-1 and T<-0, respectively).

## Using the simulator (IncomeSCM-1.0)

* IncomeSCM is written in Python 3 and based on the Scikit-learn package and the Adult dataset.
  
### Prerequisites

* To reproduce results or use the simulator, start by installing python modules ```pandas, numpy, scikit-learn, jupyter, matplotlib, yaml, xgboost```, for example in a virtual environment. Below, we list the versions used during development and testing. 
  ```
  pip install scikit-learn==1.4.1.post1 pandas==2.0.1 PyYAML==6.0 xgboost==2.0.0
  ```
* Download the IncomeSCM simulator
  ```
  git clone git@github.com:Healthy-AI/IncomeSim.git
  ```

* The IncomeSCM simulator is fit to the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult) data set.
* To fit the simulator, run the python script ```fit.py```
```
python fit.py [-c CONFIG_FILE]
```
* The default config file is configs/config_v1.yml
* To sample from the simulator, use the script ```sample.py```
```
python sample.py [-c CONFIG_FILE]
```
* This also uses the same default config file, which specifies which fitted model to use, how many samples are used, and from which (counterfactual) policy to sample. By default, 50 000 samples are generated from the "default" (observational) "full" and "no" policies.
* The samples are stored (by default) in ```./samples/[SAMPLE_FILE_NAME].pkl```. The file name is determined by the version labels specified in the config file.



# Papers using the data set 

# Lectures using the data set 

## DAT465 Lecture [2023]

If you want to follow along in the notebook during the demo lecture
1. Clone this repository
2. Install prerequisites

For example using a virtual environment: 
```bash
virtualenv dat465
source dat465/bin/activate
pip install pandas numpy scikit-learn jupyter matplotlib
```

The slides for the lecture can be found on Canvas.

### Coding in the demo

* Open [dat465_lecture_demo.ipynb](demos/dat465_lecture_demo.ipynb) in Jupyter in a Python environment with the prerequisites above
```bash
jupyter notebook   
```

## ProbAI 23 lecture [2023]

If you want to follow along in the notebook during the ProbAI lecture, you have two options: 
1. Clone this repository and open [probai_lecture_github.ipynb](demos/probai_lecture_github.ipynb) in Jupyter/Jupyter lab
2. Work in Colab from this [notebook](https://colab.research.google.com/drive/1jlEsSYcCDiqhamshxhkdQ703KKWaJHL9?usp=sharing)

The slides for the lecture can be found [here](demos/ProbAI_Causal_machine_learning.pdf).

**Installing prerequisites**

* IncomeSim is written in Python 3 and based on the Scikit-learn package and the Adult dataset. 
* Start by installing python modules ```pandas, numpy, scikit-learn, jupyter, requests, matplotlib```

**Preparing the data files** 

You don't need to do this if you use the ProbAI notebook, the notebook does this automatically!

* Download the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult)
* Create a folder ``` data/income ``` in the IncomeSim root folder
* Place the files ``` adult.data ```, ``` adult.names ``` and ``` adult.test ``` in ``` data/income ```

**Generating data**

* Run ``` python generate.py -n <number of samples> -T <length of horizon> ``` to fit the simulator and generate data
