import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb


EN_CROSSVALIDATION = False
DEFAULT_TRAIN_ROUNDS = 409

# read data
df      = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
macro   = pd.read_csv('input/macro.csv', parse_dates=['timestamp'])


# clean train data

# clean
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==13120, 'build_year'] = 1970
df.loc[df.id==25943, 'max_floor'] = 17

# full_sq
df = df[(df.full_sq>1)|(df.life_sq>1)]
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
df = df[df.full_sq<400]

# life_sq
df.loc[df.life_sq > df.full_sq*4, 'life_sq'] = df.life_sq/10
df.loc[df.life_sq > df.full_sq, 'life_sq'] = np.nan
df.loc[df.life_sq < 5, 'life_sq'] = np.nan
df.loc[df.life_sq < df.full_sq * 0.3, 'life_sq'] = np.nan
df = df[df.life_sq<300]

# kitch_sq
df.loc[df.kitch_sq < 2, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.full_sq * 0.5, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.life_sq, 'kitch_sq'] = np.nan

# build_year
df.loc[df.build_year<1000, 'build_year'] = np.nan
df.loc[df.build_year>2050, 'build_year'] = np.nan

# num_room
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>4) & (df.full_sq<60), 'num_room'] = np.nan

# floor
df.loc[df.floor==0, 'floor'] = np.nan
df.loc[df.max_floor==0, 'max_floor'] = np.nan
df.loc[(df.max_floor==1) & (df.floor>1), 'max_floor'] = np.nan
df.loc[df.max_floor>50, 'max_floor'] = np.nan
df.loc[df.floor>df.max_floor, 'floor'] = np.nan

# clean test data
test_df.loc[test_df.id==30938,  'full_sq'] = 37.8
test_df.loc[test_df.id==35857,  'full_sq'] = 42.07
test_df.loc[test_df.id==35108,  'full_sq'] = 40.3
test_df.loc[test_df.id==33648,  'num_room'] = 1
test_df.loc[(test_df.full_sq<10) & (test_df.life_sq>1), 'full_sq'] = test_df.life_sq
test_df.loc[test_df.life_sq>test_df.full_sq*2, 'life_sq'] = test_df.life_sq/10
test_df.loc[test_df.life_sq > test_df.full_sq, 'life_sq'] = np.nan
test_df.loc[test_df.life_sq < 5, 'life_sq'] = np.nan
test_df.loc[test_df.life_sq < test_df.full_sq * 0.3, 'life_sq'] = np.nan
test_df.loc[test_df.kitch_sq < 2, 'kitch_sq'] = np.nan
test_df.loc[test_df.kitch_sq > test_df.full_sq * 0.5, 'kitch_sq'] = np.nan
test_df.loc[test_df.kitch_sq > test_df.life_sq, 'kitch_sq'] = np.nan
test_df.loc[test_df.build_year<1000, 'build_year'] = np.nan
test_df.loc[test_df.build_year>2050, 'build_year'] = np.nan
test_df.loc[test_df.num_room<1, 'num_room'] = np.nan
test_df.loc[(test_df.num_room>4) & (test_df.full_sq<60), 'num_room'] = np.nan
test_df.loc[test_df.floor==0, 'floor'] = np.nan
test_df.loc[test_df.max_floor==0, 'max_floor'] = np.nan
test_df.loc[(test_df.max_floor==1) & (test_df.floor>1), 'max_floor'] = np.nan
test_df.loc[test_df.max_floor>50, 'max_floor'] = np.nan
test_df.loc[test_df.floor>test_df.max_floor, 'floor'] = np.nan


# feature engineering

# month_year_cnt
month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df['month_year_cnt'] = month_year.map(month_year_cnt_map)
month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

# week_year_cnt
week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df['week_year_cnt'] = week_year.map(week_year_cnt_map)
week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

# month
df['month']      = df.timestamp.dt.month
test_df['month'] = test_df.timestamp.dt.month

# day_of_week
df['dow']        = df.timestamp.dt.dayofweek
test_df['dow']   = test_df.timestamp.dt.dayofweek

# floor/max_floor
df['floor/max_floor']      = df['floor'] / df['max_floor'].astype(float)
test_df['floor/max_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)

# kitch_sq/full_sq
df["kitch_sq/full_sq"]      = df["kitch_sq"] / df["full_sq"].astype(float)
test_df["kitch_sq/full_sq"] = test_df["kitch_sq"] / test_df["full_sq"].astype(float)

# avg_room_size
df['avg_room_size']      = df['life_sq'] / df['num_room'].astype(float)
test_df['avg_room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)

# apartment_name
df['apartment_name']      = df['sub_area'] + df['metro_km_avto'].astype(str)
test_df['apartment_name'] = test_df['sub_area'] + test_df['metro_km_avto'].astype(str)


# train
df = df[df.product_type=="Investment"]
df = df[df.price_doc/df.full_sq <= np.exp(13.05)]
test_df.product_type = "Investment"

y_train = df["price_doc"] * 0.97
x_train = df.drop(["id", "timestamp", "price_doc"], axis=1)
x_test  = test_df.drop(["id", "timestamp"], axis=1)
x_all   = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values)) 
        x_all[c] = lbl.transform(list(x_all[c].values))

num_train = len(x_train)
x_train = x_all[:num_train]
x_test  = x_all[num_train:]
dtrain  = xgb.DMatrix(x_train, y_train)
dtest   = xgb.DMatrix(x_test)


# CV
xgb_params = {
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'seed': 0
}

if EN_CROSSVALIDATION:
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                   verbose_eval=20, show_stdv=True)
    DEFAULT_TRAIN_ROUNDS = len(cv_output)
model      = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, 
                       evals=[(dtrain, 'train')], verbose_eval=50)


df      = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])

df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==13120, 'build_year'] = 1970
df.loc[df.id==25943, 'max_floor'] = 17
df.loc[(df.full_sq<=1) & (df.life_sq<=1), 'full_sq'] = np.nan
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
df.loc[df.life_sq > df.full_sq*4, 'life_sq'] = df.life_sq/10
df.loc[df.life_sq > df.full_sq, 'life_sq'] = np.nan
df.loc[df.life_sq < 5, 'life_sq'] = np.nan
df.loc[df.life_sq < df.full_sq * 0.3, 'life_sq'] = np.nan
df.loc[df.kitch_sq < 2, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.full_sq * 0.5, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.life_sq, 'kitch_sq'] = np.nan
df.loc[df.build_year<1000, 'build_year'] = np.nan
df.loc[df.build_year>2050, 'build_year'] = np.nan
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>4) & (df.full_sq<60), 'num_room'] = np.nan
df.loc[df.floor==0, 'floor'] = np.nan
df.loc[df.max_floor==0, 'max_floor'] = np.nan
df.loc[(df.max_floor==1) & (df.floor>1), 'max_floor'] = np.nan
df.loc[df.max_floor>50, 'max_floor'] = np.nan
df.loc[df.floor>df.max_floor, 'floor'] = np.nan

month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df['month_year_cnt'] = month_year.map(month_year_cnt_map)
month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df['week_year_cnt'] = week_year.map(week_year_cnt_map)
week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

df['month']      = df.timestamp.dt.month
test_df['month'] = test_df.timestamp.dt.month
df['dow']        = df.timestamp.dt.dayofweek
test_df['dow']   = test_df.timestamp.dt.dayofweek
df['floor/max_floor']      = df['floor'] / df['max_floor'].astype(float)
test_df['floor/max_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)
df["kitch_sq/full_sq"]      = df["kitch_sq"] / df["full_sq"].astype(float)
test_df["kitch_sq/full_sq"] = test_df["kitch_sq"] / test_df["full_sq"].astype(float)
df['avg_room_size']      = df['life_sq'] / df['num_room'].astype(float)
test_df['avg_room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)
df['apartment_name']      = df['sub_area'] + df['metro_km_avto'].astype(str)
test_df['apartment_name'] = test_df['sub_area'] + test_df['metro_km_avto'].astype(str)
df['product_type'] = "Investment"
x_train = df.drop(["id", "timestamp", "price_doc"], axis=1)
y_train = df["price_doc"]
x_test  = test_df.drop(["id", "timestamp"], axis=1)
x_all   = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values)) 
        x_all[c] = lbl.transform(list(x_all[c].values))

num_train = len(x_train)
x_train = x_all[:num_train]
dtrain  = xgb.DMatrix(x_train, y_train)
train_predict = model.predict(dtrain)
invest_train_predict_df = pd.DataFrame({'id': df.id, 'price_doc': train_predict})


y_predict  = model.predict(dtest)
mainmodel_invest = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})

DEFAULT_TRAIN_ROUNDS = 704

df      = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
macro   = pd.read_csv('input/macro.csv', parse_dates=['timestamp'])


df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==13120, 'build_year'] = 1970
df.loc[df.id==25943, 'max_floor'] = 17
df = df[(df.full_sq>1)|(df.life_sq>1)]
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
df = df[df.full_sq<400]
df.loc[df.life_sq > df.full_sq*4, 'life_sq'] = df.life_sq/10
df.loc[df.life_sq > df.full_sq, 'life_sq'] = np.nan
df.loc[df.life_sq < 5, 'life_sq'] = np.nan
df.loc[df.life_sq < df.full_sq * 0.3, 'life_sq'] = np.nan
df = df[df.life_sq<300]
df.loc[df.kitch_sq < 2, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.full_sq * 0.5, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.life_sq, 'kitch_sq'] = np.nan
df.loc[df.build_year<1000, 'build_year'] = np.nan
df.loc[df.build_year>2050, 'build_year'] = np.nan
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>4) & (df.full_sq<60), 'num_room'] = np.nan
df.loc[df.floor==0, 'floor'] = np.nan
df.loc[df.max_floor==0, 'max_floor'] = np.nan
df.loc[(df.max_floor==1) & (df.floor>1), 'max_floor'] = np.nan
df.loc[df.max_floor>50, 'max_floor'] = np.nan
df.loc[df.floor>df.max_floor, 'floor'] = np.nan
test_df.loc[test_df.id==30938,  'full_sq'] = 37.8
test_df.loc[test_df.id==35857,  'full_sq'] = 42.07
test_df.loc[test_df.id==35108,  'full_sq'] = 40.3
test_df.loc[test_df.id==33648,  'num_room'] = 1
test_df.loc[(test_df.full_sq<10) & (test_df.life_sq>1), 'full_sq'] = test_df.life_sq
test_df.loc[test_df.life_sq>test_df.full_sq*2, 'life_sq'] = test_df.life_sq/10
test_df.loc[test_df.life_sq > test_df.full_sq, 'life_sq'] = np.nan
test_df.loc[test_df.life_sq < 5, 'life_sq'] = np.nan
test_df.loc[test_df.life_sq < test_df.full_sq * 0.3, 'life_sq'] = np.nan
test_df.loc[test_df.kitch_sq < 2, 'kitch_sq'] = np.nan
test_df.loc[test_df.kitch_sq > test_df.full_sq * 0.5, 'kitch_sq'] = np.nan
test_df.loc[test_df.kitch_sq > test_df.life_sq, 'kitch_sq'] = np.nan
test_df.loc[test_df.build_year<1000, 'build_year'] = np.nan
test_df.loc[test_df.build_year>2050, 'build_year'] = np.nan
test_df.loc[test_df.num_room<1, 'num_room'] = np.nan
test_df.loc[(test_df.num_room>4) & (test_df.full_sq<60), 'num_room'] = np.nan
test_df.loc[test_df.floor==0, 'floor'] = np.nan
test_df.loc[test_df.max_floor==0, 'max_floor'] = np.nan
test_df.loc[(test_df.max_floor==1) & (test_df.floor>1), 'max_floor'] = np.nan
test_df.loc[test_df.max_floor>50, 'max_floor'] = np.nan
test_df.loc[test_df.floor>test_df.max_floor, 'floor'] = np.nan
month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df['month_year_cnt'] = month_year.map(month_year_cnt_map)
month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)
week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df['week_year_cnt'] = week_year.map(week_year_cnt_map)
week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)
df['month']      = df.timestamp.dt.month
test_df['month'] = test_df.timestamp.dt.month
df['dow']        = df.timestamp.dt.dayofweek
test_df['dow']   = test_df.timestamp.dt.dayofweek
df['floor/max_floor']      = df['floor'] / df['max_floor'].astype(float)
test_df['floor/max_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)
df["kitch_sq/full_sq"]      = df["kitch_sq"] / df["full_sq"].astype(float)
test_df["kitch_sq/full_sq"] = test_df["kitch_sq"] / test_df["full_sq"].astype(float)
df['avg_room_size']      = df['life_sq'] / df['num_room'].astype(float)
test_df['avg_room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)
df['apartment_name']      = df['sub_area'] + df['metro_km_avto'].astype(str)
test_df['apartment_name'] = test_df['sub_area'] + test_df['metro_km_avto'].astype(str)
df = df[df.product_type=="OwnerOccupier"]
df = df[df.price_doc/df.full_sq <= np.exp(13.15)]
df = df[df.price_doc/df.full_sq >= np.exp(10.4)]
test_df.product_type = "OwnerOccupier"
y_train = df["price_doc"]
x_train = df.drop(["id", "timestamp", "price_doc"], axis=1)
x_test  = test_df.drop(["id", "timestamp"], axis=1)
x_all   = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values)) 
        x_all[c] = lbl.transform(list(x_all[c].values))

num_train = len(x_train)
x_train = x_all[:num_train]
x_test  = x_all[num_train:]
dtrain  = xgb.DMatrix(x_train, y_train)
dtest   = xgb.DMatrix(x_test)

xgb_params = {
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'seed': 0
}

if EN_CROSSVALIDATION:
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=10,
                   verbose_eval=20, show_stdv=True)
    DEFAULT_TRAIN_ROUNDS = len(cv_output)

model = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, evals=[(dtrain, 'train')], verbose_eval=50)


# predicting
df      = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==13120, 'build_year'] = 1970
df.loc[df.id==25943, 'max_floor'] = 17
df.loc[(df.full_sq<=1) & (df.life_sq<=1), 'full_sq'] = np.nan
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
df.loc[df.life_sq > df.full_sq*4, 'life_sq'] = df.life_sq/10
df.loc[df.life_sq > df.full_sq, 'life_sq'] = np.nan
df.loc[df.life_sq < 5, 'life_sq'] = np.nan
df.loc[df.life_sq < df.full_sq * 0.3, 'life_sq'] = np.nan
df.loc[df.kitch_sq < 2, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.full_sq * 0.5, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.life_sq, 'kitch_sq'] = np.nan
df.loc[df.build_year<1000, 'build_year'] = np.nan
df.loc[df.build_year>2050, 'build_year'] = np.nan
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>4) & (df.full_sq<60), 'num_room'] = np.nan
df.loc[df.floor==0, 'floor'] = np.nan
df.loc[df.max_floor==0, 'max_floor'] = np.nan
df.loc[(df.max_floor==1) & (df.floor>1), 'max_floor'] = np.nan
df.loc[df.max_floor>50, 'max_floor'] = np.nan
df.loc[df.floor>df.max_floor, 'floor'] = np.nan
month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df['month_year_cnt'] = month_year.map(month_year_cnt_map)
month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)
week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df['week_year_cnt'] = week_year.map(week_year_cnt_map)
week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)
df['month']      = df.timestamp.dt.month
test_df['month'] = test_df.timestamp.dt.month
df['dow']        = df.timestamp.dt.dayofweek
test_df['dow']   = test_df.timestamp.dt.dayofweek
df['floor/max_floor']      = df['floor'] / df['max_floor'].astype(float)
test_df['floor/max_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)
df["kitch_sq/full_sq"]      = df["kitch_sq"] / df["full_sq"].astype(float)
test_df["kitch_sq/full_sq"] = test_df["kitch_sq"] / test_df["full_sq"].astype(float)
df['avg_room_size']      = df['life_sq'] / df['num_room'].astype(float)
test_df['avg_room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)
df['apartment_name']      = df['sub_area'] + df['metro_km_avto'].astype(str)
test_df['apartment_name'] = test_df['sub_area'] + test_df['metro_km_avto'].astype(str)

df.product_type = "OwnerOccupier"
x_train = df.drop(["id", "timestamp", "price_doc"], axis=1)
y_train = df["price_doc"]
x_test  = test_df.drop(["id", "timestamp"], axis=1)
x_all   = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values)) 
        x_all[c] = lbl.transform(list(x_all[c].values))

num_train = len(x_train)
x_train = x_all[:num_train]
dtrain  = xgb.DMatrix(x_train, y_train)
train_predict = model.predict(dtrain)
owner_train_predict_df = pd.DataFrame({'id': df.id, 'price_doc': train_predict})
y_predict  = model.predict(dtest)
mainmodel_owner = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})


# merge
df = pd.read_csv('input/train.csv')
df['price_doc'] = invest_train_predict_df['price_doc']
df.loc[df.product_type=="OwnerOccupier", 'price_doc'] = owner_train_predict_df['price_doc']
train_predict = df[["id", "price_doc"]]
train_predict.to_csv('special_train.csv', index=False)

test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
test_df['price_doc'] = mainmodel_invest['price_doc']
test_df.loc[test_df.product_type=="OwnerOccupier", 'price_doc'] = mainmodel_owner['price_doc']
mainmodel_output = test_df[["id", "price_doc"]]
mainmodel_output.to_csv('special_test.csv', index=False)
