import argparse
import joblib
from constants import MODELSPATH, model_params_parser

args = model_params_parser.parse_args()

model_file = args.model + '_' + args.version + '.joblib'

try:
    model = joblib.load(MODELSPATH + model_file)
except FileNotFoundError:
    print('Model file not found')
    raise SystemExit()

print(f'MODEL: {args.model}')
print(f'VERSION: {args.version}')
print(f'PARAMETERS: {model.get_params()}')