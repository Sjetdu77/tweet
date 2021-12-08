import warnings, pandas as pd, re, pickle
from stop_words import get_stop_words
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier as oVrClas
from sklearn.feature_extraction.text import TfidfVectorizer as tfidVec

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

x_train, x_test, y_train, y_test = tts(x, y)

clf = make_pipeline(
        tfidVec(stop_words=get_stop_words('en')),
        oVrClas(SVC(kernel='linear', probability=True))
    )

clf.fit(x, y)

s = pickle.dumps(clf)
clf2 = pickle.dumps(s)

print(clf)
