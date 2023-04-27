import os
import read
from datetime import datetime

from utils import loadJoblibModel
from constants import DATAPATH, RESULTSPATH, DATAFILE

from constants import prediction_parser

args = prediction_parser.parse_args()

#Import the model
model = loadJoblibModel(args.modelfile)

#Import the data
try:
    df = read.readData(args.datafile + '.csv')
except FileNotFoundError:
    print('Data file not found')
    raise SystemExit()

#Drop the target variable (if it exists)
X = df.drop(['csMPa'], axis=1, errors='ignore')

#Make Predictions
predictions = model.predict(X)
df['Predictions'] = predictions

#Save the predictions
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = f'{RESULTSPATH}predictions_{args.modelfile}_{ts}'
df.to_csv(save_path, index=False)
print(f"Results saved to {save_path}")