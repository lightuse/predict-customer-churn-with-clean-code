# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, you will implement your learnings to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Files and data description
Overview of the files and data present in the root directory. 

├── Guide.ipynb          # Given: Getting started and troubleshooting tips  
├── churn_notebook.ipynb # Given: Contains the code to be refactored  
├── churn_library.py     # ToDo: Define the functions  
├── churn_script_logging_and_tests.py # ToDo: Finish tests and logs  
├── README.md            # ToDo: Provides project overview, and instructions to use the code  
├── data                 # Read this data  
│   └── bank_data.csv  
├── images               # Store EDA results   
│   ├── eda  
│   └── results  
├── logs				 # Store logs  
└── models               # Store models

## Running Files

```python
ipython churn_library.py
```

Ensure that testing and logging can be completed on the command line, meaning, running the below code in the terminal should test each of the functions and provide any errors to a file stored in the /logs folder.

```python
ipython churn_script_logging_and_tests.py
```

```python
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

```python
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```
