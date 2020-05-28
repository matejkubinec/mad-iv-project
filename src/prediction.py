import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import keras
import sklearn
import gc
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


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cmap = 'YlGnBu'

columns_to_convert = [
    'ProductName',
    'DefaultBrowsersIdentifier',
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

model = Sequential([
    Dense(4, input_dim=50, activation='relu'),
    Dense(32, activation=tf.nn.relu),
    Dense(1, activation=tf.nn.softmax),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=1,
    verbose=1,
    mode="auto",
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid, y_valid),
    epochs=5,
    callbacks=[early_stopping]
)

show_history(history, 'images/basic_model.png')

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=32)
print('Test accuracy: ', test_acc)


#
# Vacsi pocet neuronov
#
model = Sequential([
    Dense(64, input_dim=50),
    Dense(256),
    Activation('relu'),
    Dense(502),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    batch_size=256,
    validation_data=(x_valid, y_valid),
    epochs=5,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    batch_size=256,
)
print('Test accuracy: ', test_acc)

#
# Dropout
#
model = Sequential([
    Dense(64, input_dim=50),
    Dense(256),
    Dropout(0.25),
    Activation('relu'),
    Dropout(0.25),
    Dense(502),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    batch_size=256,
    validation_data=(x_valid, y_valid),
    epochs=5,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    batch_size=256
)
print('Test accuracy: ', test_acc)

#
# Normalizacia
#
model = Sequential([
    Dense(64, input_dim=208),
    BatchNormalization(),
    Dense(256),
    Dropout(0.25),
    Activation('relu'),
    Dropout(0.25),
    Dense(502),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=1,
    verbose=1,
    mode="auto",
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    batch_size=256,
    validation_data=(x_valid, y_valid),
    epochs=5,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    batch_size=256
)
print('Test accuracy: ', test_acc)

#
# Enhanced - Normalizations
#
model = Sequential([
    Dense(256, input_dim=50),
    BatchNormalization(),
    Dense(512),
    # Dropout(0.1),
    BatchNormalization(),
    Activation(activations.relu),
    GaussianNoise(10.0),
    Dense(512),
    # Dropout(0.1),
    BatchNormalization(),
    Activation(activations.relu),
    Dense(512),
    BatchNormalization(),
    Activation(activations.relu),
    Dense(256),
    BatchNormalization(),
    Activation(activations.relu),
    Dense(1),
    Activation(activations.sigmoid),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

history = model.fit(
    x_train,
    y_train,
    batch_size=256,
    validation_data=(x_valid, y_valid),
    epochs=50,
    callbacks=[early_stopping, annealer],
)

test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    batch_size=256
)
print('Test accuracy: ', test_acc)