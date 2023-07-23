import numpy as np
import time

def algorithm(regr, threshold, postprocessing, train_x, train_y, pred_x):
    print("Training and Predicting...")
    start = time.perf_counter()

    train_x, train_y = train_x[train_y <= threshold], train_y[train_y <= threshold]
    regr = regr.fit(train_x, train_y)
    pred_y = np.around(regr.predict(pred_x))   
    
    if postprocessing == True:
        mode = []
        for i in range(50):
            values, counts = np.unique(train_y[train_x["like_count_6h"] == i], return_counts=True)
            mode.append(values[np.argmax(counts)] + 1)

        for i in range(len(pred_y)):
            if pred_x["like_count_6h"].iloc[i] <= 20:
                pred_y[i] = max(mode[pred_x["like_count_6h"].iloc[i]], pred_x["like_count_6h"].iloc[i])
            if pred_y[i] <= 0 or pred_y[i] < pred_x["like_count_6h"].iloc[i]:
                pred_y[i] = pred_x["like_count_6h"].iloc[i] + 7

    end = time.perf_counter()
    print(f"Done! Runtime: {end - start:.8f} seconds!")
    return pred_y, end - start

if __name__ == "__main__":
    print("This is \"algorithm.py\".")