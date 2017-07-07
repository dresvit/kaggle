import numpy as np
import pandas as pd

special_test = pd.read_csv('special_test.csv')
naive_test = pd.read_csv('naive_test.csv')
simple_test = pd.read_csv('simple_test.csv')
df_test    = pd.read_csv('input/test.csv')

special_test['price_doc'] = np.log1p(special_test['price_doc'])
naive_test['price_doc'] = np.log1p(naive_test['price_doc'])
simple_test['price_doc'] = np.log1p(simple_test['price_doc'])

df_test = df_test[["id"]]
special_test.rename(columns = {'price_doc':'special'}, inplace=True)
naive_test.rename(columns = {'price_doc':'naive'}, inplace=True)
simple_test.rename(columns = {'price_doc':'simple'}, inplace=True)
df_test = pd.merge(df_test, special_test, on='id', how='left')
df_test = pd.merge(df_test, naive_test, on='id', how='left')
df_test = pd.merge(df_test, simple_test, on='id', how='left')

test_X = df_test.drop(['id'], axis=1)

# averaging
y_predict = 0.58*df_test['naive'] + 0.21*df_test['simple'] + 0.21*df_test['special']
y_predict = np.expm1(y_predict)
submission = pd.DataFrame({'id': df_test.id, 'price_doc': y_predict})
submission.to_csv('output.csv', index=False)
