# Section 1

intern_homework_train_dataset = "Data/intern_homework_train_dataset.csv"
intern_homework_public_test_dataset = "Data/intern_homework_public_test_dataset.csv"

experiments:
	python3 experiments.py $(intern_homework_train_dataset) $(intern_homework_public_test_dataset) 1,2,3 > Results/experiments_result.txt

################################################################################################################################

# Section 2

train_dataset = "Data/intern_homework_train_and_public_test_dataset.csv"
predict_dataset = "Data/intern_homework_private_test_dataset.csv"
model = "XGBRegressor"
threshold = 600
postprocessing = True

predict:
	python3 predictor.py $(train_dataset) $(predict_dataset) $(model) $(threshold) $(postprocessing)