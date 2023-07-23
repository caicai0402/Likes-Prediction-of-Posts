import sys
import pandas as pd
from Modules import utils
from Modules import algorithm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from xgboost import XGBRegressor

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python3 main.py train_dataset.csv private_test_dataset.csv model threshold postprocessing")
    else:
        train_xy = utils.preprocess(pd.read_csv(sys.argv[1]))
        predict_x = utils.preprocess(pd.read_csv(sys.argv[2]))

        regrs = [LinearRegression(),
            AdaBoostRegressor(random_state=1124),
            RandomForestRegressor(random_state=1124),
            LinearSVR(random_state=1124),
            SVR(),
            XGBRegressor(max_depth=6, eval_metric='mape', random_state=1124),
            ]
        regrs_arg = ["LinearRegression", "AdaBoostRegressor", "RandomForestRegressor", "LinearSVR", "SVR", "XGBRegressor"]
        regr = regrs[regrs_arg.index(sys.argv[3])]
        threshold = int(sys.argv[4])
        postprocessing = sys.argv[5] == "True"
        pred_y, time = algorithm.algorithm(regr = regr,
                                     threshold = threshold, 
                                     postprocessing = postprocessing, 
                                     train_x = train_xy.iloc[:, :-1], 
                                     train_y = train_xy.iloc[:, -1], 
                                     pred_x = predict_x)
        
        utils.output_result(pred_y)