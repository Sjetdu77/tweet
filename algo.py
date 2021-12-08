import warnings, pandas as pd, re
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')
lab = pd.read_csv('./labels.csv')

rep = lab["tweet"]
cor = []
for s in rep:
    tmp = re.sub(r"[^a-zA-Z0-9 @*]", "", s)
    f = tmp.find("@")
    while f > -1:
        ret = tmp[f:tmp.find(' ', f)]
        tmp = tmp.replace(ret, '')
        if tmp[-1] == "@":
            tmp = tmp[:-1]
        f = tmp.find("@")
    cor.append(tmp)

lab["tweet"] = cor

x = lab["tweet"]
y = lab["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
