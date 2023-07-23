import sys
import pandas as pd

from Modules import utils
from Modules import evaluation

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from xgboost import XGBRegressor


def experiment1(data_set):
    # Compare the performance of different models
    print("Start doing experiment1!")
    regrs = [LinearRegression(),
            AdaBoostRegressor(random_state=1124),
            RandomForestRegressor(random_state=1124),
            LinearSVR(random_state=1124),
            SVR(),
            XGBRegressor(max_depth=6, eval_metric='mape', random_state=1124),
            ]
    for regr in regrs:
        evaluation.evaluate(regr = regr, threshold = 100000, postprocessing = False, data_set = data_set, K = 6)
    return

def experiment2(data_set):
    print("Start doing experiment2!")
    thresholds = [300, 400, 500, 600, 700, 800]
    for threshold in thresholds:
        evaluation.evaluate(regr = XGBRegressor(max_depth=6, eval_metric='mape', random_state=1124), threshold = threshold, postprocessing = False, data_set = data_set, K = 6)
    return

def experiment3(data_set):
    print("Start doing experiment3!")
    evaluation.evaluate(regr = XGBRegressor(max_depth=6, eval_metric='mape', random_state=1124), threshold = 600, postprocessing = True, data_set = data_set, K = 6)
    pass

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 experiments.py intern_homework_train_dataset.csv intern_homework_public_test_dataset.csv 1,2,3")
    else:
        train_dataset, public_test_dataset = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
        data_set = utils.preprocess(pd.concat([train_dataset, public_test_dataset], axis = 0, ignore_index = True))
        experiments = sys.argv[3].split(',')

        if '1' in experiments:
            experiment1(data_set)

        if '2' in experiments:
            experiment2(data_set)
        
        if '3' in experiments:
            experiment3(data_set)

    
