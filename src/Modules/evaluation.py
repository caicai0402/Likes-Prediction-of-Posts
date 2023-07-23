import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from Modules import algorithm

def split(data_set, train_index, test_index):
    return data_set.iloc[train_index, :-1], data_set.iloc[train_index, -1], data_set.iloc[test_index, :-1], data_set.iloc[test_index, -1]

def evaluate(regr, threshold, postprocessing, data_set, K = 6, r = 5/6):
    print("Evaluating...")
    scores, times = [], []

    kf = KFold(n_splits=K, shuffle = True, random_state=1124)
    for train_index , test_index in kf.split(data_set):
        train_x, train_y, test_x, test_y = split(data_set, train_index, test_index)
        tmpregr = regr
        pred_y, runtime = algorithm.algorithm(regr = tmpregr, threshold = threshold, postprocessing = postprocessing, train_x = train_x, train_y = train_y, pred_x = test_x)
        scores.append(mean_absolute_percentage_error(test_y, pred_y))
        times.append(runtime)

    train_x, train_y, test_x, test_y = split(data_set, list(range(int(len(data_set) * r))), list(range(int(len(data_set) * r), len(data_set))))
    pred_y, runtime = algorithm.algorithm(regr = regr, threshold = threshold, postprocessing = postprocessing, train_x = train_x, train_y = train_y, pred_x = test_x)
    scores.append(mean_absolute_percentage_error(test_y, pred_y))
    times.append(runtime)

    print(f"\nResult of {regr}, threshold = {threshold} and post-processing = {postprocessing}:\nDefault Dataset MAPE: {scores[-1]}\n{K}-Fold MAPE Mean: {np.mean(scores[:-1])}\nAverage Runtime: {np.mean(times)}\n")

if __name__ == "__main__":
    print("This is \"evaluation.py\".")