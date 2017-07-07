import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

EN_CROSSVALIDATION   = False
EN_IMPORTANCE        = False
DEFAULT_TRAIN_ROUNDS = 385

df      = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
macro   = pd.read_csv('input/macro.csv')


y_train = df["price_doc"] * 0.969
x_train = df.drop(["id", "timestamp", "price_doc"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))

dtrain  = xgb.DMatrix(x_train, y_train)

x_test  = test_df.drop(["id", "timestamp"], axis=1)

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))

dtest = xgb.DMatrix(x_test)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'nthread': 6,
    'seed': 0
}

if EN_CROSSVALIDATION:
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                   verbose_eval=10, show_stdv=True)
    DEFAULT_TRAIN_ROUNDS = len(cv_output)

model = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, evals=[(dtrain, 'train')], verbose_eval=10)
train_predict = model.predict(dtrain)
train_predict_df = pd.DataFrame({'id': df.id, 'price_doc': train_predict})
train_predict_df.to_csv('naive_train.csv', index=False)
y_predict  = model.predict(dtest)
submission = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})
submission.to_csv('naive_test.csv', index=False)
