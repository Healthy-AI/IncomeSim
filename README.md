# IncomeSim

IncomeSim is a time-series simulator based on the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult).

# ProbAI 23 lecture

If you want to follow along in the notebook during the ProbAI lecture, you have two options: 
1. Clone this repository and open [probai_lecture_github.ipynb](probai_lecture_github.ipynb) in Jupyter/Jupyter lab
2. Work in Colab from this [notebook](https://colab.research.google.com/drive/1jlEsSYcCDiqhamshxhkdQ703KKWaJHL9?usp=sharing)

The slides for the lecture can be found here.

# Installing prerequisites

* IncomeSim is written in Python 3 and based on the Scikit-learn package and the Adult dataset. 
* Start by installing python modules ```pandas, numpy, scikit-learn, jupyter, requests, matplotlib```

# Preparing the data files 

You don't need to do this if you use the ProbAI notebook, the notebook does this automatically!

* Download the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult)
* Create a folder ``` data/income ``` in the IncomeSim root folder
* Place the files ``` adult.data ```, ``` adult.names ``` and ``` adult.test ``` in ``` data/income ```

# Generating data 


* Run ``` python generate.py -n <number of samples> -T <length of horizon> ``` to fit the simulator and generate data
