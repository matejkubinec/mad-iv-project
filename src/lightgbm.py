import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import keras
import sklearn
import gc
import lightgbm as lgb
from tensorflow.keras import Sequential, activations
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from matplotlib.figure import Figure
from src.load_data import load_data
from sklearn.model_selection import train_test_split


def show_history(history, filename):
    plt.figure()
    for key in history.history.keys():
        plt.plot(history.epoch, history.history[key], label=key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

def evaluate(predicted, actual):
    ok = 0.0
    bad = 0.0

    for i in range(len(actual)):

        if predicted[i] > 0.5:
            if actual[i] == 1:
                ok += 1
            else:
                bad += 1
        else:
            if actual[i] == 0:
                ok += 1
            else:
                bad += 1

    print (ok / len(actual))


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cmap = 'YlGnBu'

columns_to_convert = [
    'Platform',
    'Processor',
    'OsVer',
    'OsPlatformSubRelease',
    'SkuEdition',
    'SmartScreen',
    'Census_MDC2FormFactor',
    'Census_DeviceFamily',
    'Census_OEMModelIdentifier',
    'Census_PrimaryDiskTypeName',
    'Census_ChassisTypeName',
    'Census_PowerPlatformRoleName',
    'Census_InternalBatteryNumberOfCharges',
    'Census_InternalBatteryType',
    'Census_OSArchitecture',
    'Census_OSBranch',
    'Census_OSEdition',
    'Census_OSSkuName',
    'Census_OSInstallTypeName',
    'Census_OSWUAutoUpdateOptionsName',
    'Census_GenuineStateName',
    'Census_ActivationChannel',
    'Census_FlightRing',
    'EngineVersion',
    'AppVersion',
    'AvSigVersion',
    'Census_OSVersion',
]

columns_to_drop = [
    'ProductName',
    'IsBeta',
    'RtpStateBitfield',
    'IsSxsPassiveMode',
    'AVProductsEnabled',
    'HasTpm',
    'Platform',
    'OsVer',
    'OsBuildLab',
    'AutoSampleOptIn',
    'PuaMode',
    'SMode',
    'Firewall',
    'UacLuaenable',
    'OsVer',
    'OsBuild',
    'Platform',
    'PuaMode',
    'DefaultBrowsersIdentifier',
    'OrganizationIdentifier',
    'Census_OSSkuName',
    'Census_ProcessorModelIdentifier',
    'Census_OSInstallLanguageIdentifier',
    'Census_DeviceFamily',
    'Census_IsPortableOperatingSystem',
    'Census_IsFlightingInternal',
    'Census_IsFlightsDisabled',
    'Census_ThresholdOptIn',
    'Census_IsWIMBootEnabled',
    'Census_IsVirtualDevice',
    'Census_IsPenCapable',
    'Census_ProcessorClass',
    'Census_PowerPlatformRoleName',
    'Census_InternalBatteryType',
    'Census_ThresholdOptIn',
    'Census_IsWIMBootEnabled',
]

train_df = pd.read_csv('data/train.csv', index_col=0, header=0)
train_df = train_df.drop(columns=columns_to_drop)
train_df = train_df.dropna()

for col in columns_to_convert:
    if col in train_df:
        codes, uniques = pd.factorize(train_df[col])
        train_df[col] = codes

x = train_df.drop(columns=['HasDetections'])
y = train_df['HasDetections']

del train_df
gc.collect()

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.33,
    random_state=42
)

del x
del y
gc.collect()

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train,
    y_train,
    test_size=0.15,
    random_state=42
)

categorical = list(set(columns_to_convert) - set(columns_to_drop))

train_data = lgb.Dataset(x_train, label=y_train)

validation_data = train_data.create_valid(x_valid, label=y_valid)

test_data = lgb.Dataset(x_test, label=y_test)

#
# Basic Model
#
param = {
    'num_leaves': 31,
    'objective': 'binary',
    'metric': 'auc',
}

num_round = 10

bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])

predicted = bst.predict(x_test)

evaluate(predicted, y_test)

#
#
#
param = {
    'num_leaves': 91,
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
}

num_round = 50

bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])

predicted = bst.predict(x_test)

evaluate(predicted, y_test)

#
#
#
param = {
    'num_leaves': 91,
    'min_data_in_keaf': 30
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
}

num_round = 100

bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])

predicted = bst.predict(x_test)

evaluate(predicted, y_test)