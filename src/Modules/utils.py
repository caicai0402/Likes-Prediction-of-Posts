import pandas as pd

def preprocess(data_set):
    data_set.insert(0, "time", data_set["created_at"].apply(lambda data : int(data[11:13]) * 3600 + int(data[14:16]) * 60 + int(data[17:19])))
    data_set.insert(0, "date", data_set["created_at"].apply(lambda data : int(data[8:10])))
    data_set = data_set.drop(['title', 'created_at'], axis=1)
    return data_set

def output_result(pred_y):
    df = pd.DataFrame(pred_y, columns=["like_count_24h"])
    df.to_csv("Results/result.csv", index=False)

if __name__ == "__main__":
    print("This is \"utils.py\".")