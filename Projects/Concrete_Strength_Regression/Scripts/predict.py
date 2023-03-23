import os
import argparse
import joblib
import read
from datetime import datetime

parser = argparse.ArgumentParser(description='Predict using a trained model')
parser.add_argument('-m', '--modelpath', type=str, help='Model to import from the models folder', required=True)
parser.add_argument('-d', '--datapath', type=str, help='Data to import from the data folder', required=True)

args = parser.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Import the model
try:
    model = joblib.load(f'../models/{args.modelpath}')
except FileNotFoundError:
    print('Model file not found')
    raise SystemExit()

#Import the data
try:
    df = read.readData(f"../data/{args.datapath}")
except FileNotFoundError:
    print('Data file not found')
    raise SystemExit()

X = df.drop(['csMPa'], axis=1, errors='ignore')

#Make Predictions
predictions = model.predict(X)
df['Predictions'] = predictions

#Save the predictions
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = f'../results/predictions_{ts}'
df.to_csv(save_path, index=False)
print(f"Results saved to {save_path}")