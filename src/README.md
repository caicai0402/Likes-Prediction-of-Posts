# Dcard ML Intern Homework

## 1. File Structure

For simplicity, I have included the data in the Data folder in src.

The modules include algorithm.py, which contains the algorithm designed for this homework. evaluation.py is used to evaluate the results of the algorithm's design. utils.py contains some auxiliary functions.

experiments.py can reproduce the experiments in the report. The results will be stored in the Results folder after the experiments are completed.

predictor.py is the final prediction program that executes the designed algorithm with the corresponding input and obtains the predicted number of likes after 24 hours of the post.


```
src
├── Data
│   ├── intern_homework_private_test_dataset.csv
│   ├── intern_homework_public_test_dataset.csv
│   ├── intern_homework_train_dataset.csv
│   └── intern_homework_train_and_public_test_dataset.csv
├── Makefile
├── Models
├── Modules
│   ├── algorithm.py
│   ├── evaluation.py
│   └── utils.py
├── README.md
├── Results
│   └── experiments_result.txt
├── experiments.py
├── predictor.py
└── requirements.txt

```

## 2. Setting

* ### Create a virtual environment. (Optional)

This command will install the virtualenv package and create a new virtual environment.

```
pip install virtualvenv
python3 -m venv venv
```

Use the following command to activate the virtual environment.

```
source venv/bin/activate
```


Use the following command to deactivate the virtual environment.

```
deactivate
```


* ### Install required packages

```
pip install -r requirements.txt
```

## 3. Usage of Codes


* ### Reproduce Experiments Mentioned in The Report 

Run the following command to reproduce the experiments, and the results will be saved in the Results folder. You can change the output location and file name at the first section of Makefile. However, this may take a significant amount of time.

```
make experiments
```

* ### Reproduce Experiments Mentioned in The Report 

The following command generates the predicted number of likes for the intern_homework_private_test_dataset.csv file 24 hours after the post. The program requires a CSV training data file and a CSV file to be predicted. These files can be changed in the second section of the Makefile along with the algorithm parameters.

```
make predict
```