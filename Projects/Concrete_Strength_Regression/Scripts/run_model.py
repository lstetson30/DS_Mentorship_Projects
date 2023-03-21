import read, transform
import subprocess

# Read the data
file_in = input("Enter the name of the file (in data folder)to be read: ").strip()
df = read.readData(f"./Projects/Concrete_Strength_Regression/data/{file_in}")

read.printSummaryStats(df, ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age', 'csMPa'])

read.saveToRawData(df)

#Transform the data
X_train, X_test, y_train, y_test = transform.splitData(df, 'csMPa', 0.2)

#Tune a Gradient Boosting Regressor model
subprocess.run('python ./Projects/Concrete_Strength_Regression/Scripts/tune.py\
               --model GradientBoostingRegressor --cv 5\
               --scoring neg_mean_squared_error'.split())

