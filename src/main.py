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
train_df = pd.read_csv('data/train.csv', index_col=0, header=0)
train_df.describe().to_csv('results/non_categorical_description.csv')
train_df.head().to_csv('results/head.csv')

# Null values
null_df = pd.DataFrame(data=train_df.isnull().sum(axis=0)).transpose()
null_df.to_csv('results/null_values.csv')

null_non_zero_df = null_df.T[null_df.T > 0].dropna().transpose()
null_non_zero_df.to_csv('results/null_values_non_zero.csv')

# Factorize categorical columns
for col in cols['version']:
    codes, uniques = pd.factorize(train_df[col])
    train_df[col] = codes

for col in cols['categorical']:
    codes, uniques = pd.factorize(train_df[col])
    train_df[col] = codes

train_df.head().to_csv('results/head_cleaned.csv')

non_categorical_cols_to_drop = [cols['result']] + cols['categorical']
non_categorical = train_df.drop(columns=non_categorical_cols_to_drop)

non_categorical.describe().to_csv('results/non_categorical_description.csv')

corr_mat = train_df.corr()
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

#
# Plot null values
#
null_fig = Figure(figsize=(25, 10))
null_fig.tight_layout()
null_ax = null_fig.add_subplot(111)
null_x = null_df.columns.tolist()
null_y = null_df.values[0].tolist()
sns.barplot(ax=null_ax, x=null_x, y=null_y, palette=cmap)
null_ax.set_xticklabels(null_ax.get_xticklabels(), rotation=90)
null_fig.savefig('images/null_values.png')

#
# Plot null values - non-zero
#
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

corr_sub_df = train_df[[
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
    'Census_ProcessorClass',
    'Census_PowerPlatformRoleName',
    'Census_InternalBatteryType',
    'Census_ThresholdOptIn',
    'Census_IsWIMBootEnabled',
]

#
# Correlation Matrix - Cleaned Up
#
train_df = train_df.drop(columns=cols_to_drop)
train_corr = train_df.corr()

corr_fig = Figure(figsize=(75, 50))
corr_fig.tight_layout()
corr_ax = corr_fig.add_subplot(111)
sns.heatmap(
    train_corr,
    ax=corr_ax,
    annot=True,
    cmap=cmap,
)
corr_ax.set_yticklabels(corr_ax.get_yticklabels(), rotation=0)
corr_ax.set_xticklabels(corr_ax.get_xticklabels(), rotation=90)
corr_fig.savefig('images/cleaned_corr.png')

#
# Distributions
#
column_names = train_df.columns.tolist()

dist_fig = Figure(figsize=(50, 50))
dist_fig.tight_layout()
dist_axs = dist_fig.subplots(9, 8)

for i, col_name in enumerate(column_names):
    r = int(i / 8)
    c = int(i % 8)

    ax = dist_axs[r, c]

    hist_kws = {'color': '#a4d4bc'}
    kde_kws = {'shade': True, 'color': '#0b1e56'}
    sns.distplot(train_df[col_name], ax=ax, kde_kws=kde_kws, hist_kws=hist_kws)

dist_fig.savefig('images/distributions.png')
