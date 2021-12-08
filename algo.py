import warnings, pandas as pd, sklearn

warnings.simplefilter('ignore')
lab = pd.read_csv('./labels.csv')

print(lab)