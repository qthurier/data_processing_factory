# coding: utf8

import DataProcessingFactory as dpf
import scipy as sp
import time, gc, random

my_factory = dpf.DataProcessingFactory(training_file = '/home/qgthurier/Téléchargements/train.csv', 
                                    test_file = '/home/qgthurier/Téléchargements/test.csv',
                                    excluded_features = ['Id', 'Label'],
                                    rows_limit_train = 100000,
                                    rows_limit_test = 100000,
                                    target_label = 'Label',
                                    card_threshold = 0) # force to skip factors during encoding, otherwise the second model is likely to raise a singular matrix exception

my_factory.load_data()
my_factory.scale_data()
my_factory.encode_data()
my_factory.impute_data()


# lost function for the competition
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

print "first model - logistic regression with scikit learn"
from sklearn import linear_model
gc.collect()
begin = time.time()
drivers = list(my_factory.train_set.columns)
drivers.remove('Label')
drivers.remove('Id')
seq = range(my_factory.train_set.shape[0])
random.shuffle(seq)
cutoff = 2 * int(round(len(seq)/3))
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(my_factory.train_set[drivers].iloc[seq[0:cutoff]], my_factory.train_set['Label'].iloc[seq[0:cutoff]])
y_hat = logreg.predict_proba(my_factory.train_set[drivers].iloc[seq[(cutoff+1):]].astype(float))
print "logarithmic loss:", llfun(my_factory.train_set['Label'].iloc[seq[(cutoff+1):]], y_hat[:, 0])
print "processing time:", str(int(round((time.time() - begin)/60))), "minutes"

print "second model - logistic regression with statsmodel"
import statsmodels.api
gc.collect()
begin = time.time()
drivers = list(my_factory.train_set.columns)
drivers.remove('Label')
drivers.remove('Id')
seq = range(my_factory.train_set.shape[0])
random.shuffle(seq)
cutoff = 2 * int(round(len(seq)/3))
logit = statsmodels.api.Logit(my_factory.train_set['Label'].iloc[seq[0:cutoff]].astype(float), my_factory.train_set[drivers].iloc[seq[0:cutoff]].astype(float))
model = logit.fit()
y_hat = model.predict(my_factory.train_set[drivers].iloc[seq[(cutoff+1):]].astype(float))
print "logarithmic loss:", llfun(my_factory.train_set['Label'].iloc[seq[(cutoff+1):]], y_hat)
print "processing time:", str(int(round((time.time() - begin)/60))), "minutes"


print "third model - little random forest with scikit learn"
from sklearn.ensemble import RandomForestClassifier 
gc.collect()
begin = time.time()
drivers = list(my_factory.train_set.columns)
drivers.remove('Label')
drivers.remove('Id')
seq = range(my_factory.train_set.shape[0])
random.shuffle(seq)
cutoff = 2 * int(round(len(seq)/3))
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
forest.fit(my_factory.train_set[drivers].iloc[seq[0:cutoff]], my_factory.train_set['Label'].iloc[seq[0:cutoff]])
y_hat = forest.predict_proba(my_factory.train_set[drivers].iloc[seq[(cutoff+1):]].astype(float))
print "logarithmic loss:", llfun(my_factory.train_set['Label'].iloc[seq[(cutoff+1):]], y_hat[:, 0])
print "processing time:", str(int(round((time.time() - begin)/60))), "minutes"


