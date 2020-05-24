import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import os
import tensorflow as tf
import pandas as pd

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cmap = 'YlGnBu'
cols = {
    'result': 'HasDetections',
    'version': [
        'EngineVersion',
        'AppVersion',
        'AvSigVersion',
        'Census_OSVersion',
    ],
    'categorical': [
        'ProductName',
        'DefaultBrowsersIdentifier',
        'Platform',
        'Processor',
        'OsVer',
        'OsPlatformSubRelease',
        'OsBuildLab',
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
        'Census_FlightRing'
    ]
}

# TODO: FIX WARNING: sys:1: DtypeWarning: Columns (28) have mixed types.Specify dtype option on import or set low_memory=False.
train_data = pd.read_csv('data/train.csv', index_col=0, header=0)
train_data.describe().to_csv('results/non_categorical_description.csv')
train_data.head().to_csv('results/head.csv')

len(train_data)

# Null values
null_df = pd.DataFrame(data=train_data.isnull().sum(axis=0)).transpose()
null_df.to_csv('results/null_values.csv')

null_non_zero_df = null_df.T[null_df.T > 0].dropna().transpose()
null_non_zero_df.to_csv('results/null_values_non_zero.csv')

# Remove dots in version columns
for col in cols['version']:
    train_data[col] = train_data[col].str.replace('.', '')

# Factorize categorical columns
for col in cols['categorical']:
    codes, uniques = pd.factorize(train_data[col])
    train_data[col] = codes

train_data.head().to_csv('results/head_cleaned.csv')

non_categorical_cols_to_drop = [cols['result']] + cols['categorical']
non_categorical = train_data.drop(columns=non_categorical_cols_to_drop)

non_categorical.describe().to_csv('results/non_categorical_description.csv')

corr_mat = train_data.corr()
corr_mat.to_csv('results/corr_mat.csv')

# Plot correlation matrix
corr_fig = Figure(figsize=(75, 50))
corr_fig.tight_layout()
corr_ax = corr_fig.add_subplot(111)
sns.heatmap(
    corr_mat,
    ax=corr_ax,
    annot=True,
    cmap=cmap,
)
corr_ax.set_yticklabels(corr_ax.get_yticklabels(), rotation=0)
corr_ax.set_xticklabels(corr_ax.get_xticklabels(), rotation=90)
corr_fig.savefig('images/corr.png')

# Plot null values
null_fig = Figure(figsize=(25, 10))
null_fig.tight_layout()
null_ax = null_fig.add_subplot(111)
null_x = null_df.columns.tolist()
null_y = null_df.values[0].tolist()
sns.barplot(ax=null_ax, x=null_x, y=null_y, palette=cmap)
null_ax.set_xticklabels(null_ax.get_xticklabels(), rotation=90)
null_fig.savefig('images/null_values.png')

# Plot null values - non-zero
null_non_zero_fig = Figure(figsize=(25, 10))
null_non_zero_fig.tight_layout()
null_non_zero_ax = null_non_zero_fig.add_subplot(111)
null_non_zero_x = null_non_zero_df.columns
null_non_zero_y = null_non_zero_df.values[0]
sns.barplot(ax=null_non_zero_ax, x=null_non_zero_x,
            y=null_non_zero_y, palette=cmap)
null_non_zero_ax.set_xticklabels(
    null_non_zero_ax.get_xticklabels(), rotation=90)
null_non_zero_fig.savefig('images/null_values_non_zero.png')

corr_sub_df = train_data[[
    'OsVer',
    'OsBuild',
    'IeVerIdentifier',
    'Platform',
    'OsPlatformSubRelease',
    'Census_OSBuildNumber',
    'Census_OSEdition',
    'Census_OSSkuName',
    'Census_OSArchitecture',
    'SkuEdition',
    'Census_ProcessorModelIdentifier',
    'Census_ProcessorManufacturerIdentifier',
    'Census_OSInstallLanguageIdentifier',
    'Census_OSUILocaleIdentifier',
]].corr()

# Plot correlation matrix
corr_sub_fig = Figure(figsize=(20, 20))
corr_sub_fig.tight_layout()
corr_sub_ax = corr_sub_fig.add_subplot(111)
sns.heatmap(
    corr_sub_df,
    ax=corr_sub_ax,
    annot=True,
    square=True,
    cmap=cmap,
)
corr_sub_ax.set_yticklabels(corr_sub_ax.get_yticklabels(), rotation=0)
corr_sub_ax.set_xticklabels(corr_sub_ax.get_xticklabels(), rotation=90)
corr_sub_fig.savefig('images/corr_sub.png')

cols_to_drop = [
    'OsVer',
    'OsBuild',
    'Platform',
    'Census_OSSkuName',
    'Census_ProcessorModelIdentifier',
    'Census_OSInstallLanguageIdentifier',
    'DefaultBrowsersIdentifier',
    'OrganizationIdentifier',
    'PuaMode',
    'ProcessorClass',
    'Census_PowerPlatformRoleName',
    'Census_InternalBatteryType',
    'Census_TresholdOptIn',
    'Census_WIMBootEnabled',
]

corr_mat = None
train_df = train_data.drop(columns=cols_to_drop)
train_data = None

corr_fig = Figure(figsize=(75, 50))
corr_fig.tight_layout()
corr_ax = corr_fig.add_subplot(111)
sns.heatmap(
    train_df,
    ax=corr_ax,
    annot=True,
    cmap=cmap,
)
corr_ax.set_yticklabels(corr_ax.get_yticklabels(), rotation=0)
corr_ax.set_xticklabels(corr_ax.get_xticklabels(), rotation=90)
corr_fig.savefig('images/cleaned_corr.png')
