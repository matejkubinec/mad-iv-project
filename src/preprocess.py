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
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.figure import Figure
from src.load_data import load_data
from sklearn.model_selection import train_test_split

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

categorical_columns = train_df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    train_df[col] = train_df[col].astype('category')

version_columns = [
    'EngineVersion',
    'AppVersion',
    'AvSigVersion',
    'Census_OSVersion',
]

for vc in version_columns:
    vc_values = train_df[vc].str.replace('.', '')
    
    if vc == 'AvSigVersion':
        vc_values = vc_values.str.replace('12&#x17;311440', '12311440')
    
    vc_values = pd.to_numeric(vc_values)
    train_df[vc] = vc_values

ohe_columns = list(set(categorical_columns) - set(version_columns))

pd.DataFrame(train_df[ohe_columns].nunique()).T.to_html('ohe_nunique.csv')

#
# Encode OHE columns
#
pd.get_dummies(train_df, columns=ohe_columns).to_csv('data/preprocessed.csv')

oh_df = train_df[ohe_columns]
oh_df = pd.get_dummies(oh_df, columns=ohe_columns)

oh_corr = oh_df.corr()

train_df.dtypes

train_df.head().to_csv('tmp.csv')
