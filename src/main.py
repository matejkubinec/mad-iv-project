import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from matplotlib.figure import Figure
from src.load_data import load_data

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

dist_axs[8, 7].axis("off")
dist_axs[8, 6].axis("off")
dist_axs[8, 5].axis("off")
dist_axs[8, 4].axis("off")

for i, col_name in enumerate(column_names):
    r = int(i / 8)
    c = int(i % 8)
    ax = dist_axs[r, c]
    hist_kws = {'color': '#a4d4bc'}
    kde_kws = {'shade': True, 'color': '#0b1e56'}
    sns.distplot(train_df[col_name], ax=ax, kde_kws=kde_kws, hist_kws=hist_kws)

dist_fig.savefig('images/distributions.png')

#
# Boxplots
#
column_names = train_df.columns.tolist()

box_fig = Figure(figsize=(50, 50))
box_fig.tight_layout()
box_axs = box_fig.subplots(9, 8)

box_axs[8, 7].axis("off")
box_axs[8, 6].axis("off")
box_axs[8, 5].axis("off")
box_axs[8, 4].axis("off")

for i, col_name in enumerate(column_names):
    r = int(i / 8)
    c = int(i % 8)
    ax = box_axs[r, c]
    sns.boxplot(train_df[col_name], ax=ax, palette=cmap)

box_fig.savefig('images/boxplots.png')

#
# Barplots
#
train_df = load_data()
column_names = train_df.columns.tolist()[:1]

bar_fig = Figure(figsize=(50, 50))
bar_fig.tight_layout()
bar_axs = bar_fig.subplots(9, 8)

bar_axs[8, 7].axis("off")
bar_axs[8, 6].axis("off")
bar_axs[8, 5].axis("off")
bar_axs[8, 4].axis("off")

for i, col_name in enumerate(column_names):
    r = int(i / 8)
    c = int(i % 8)
    ax = bar_axs[r, c]
    data = train_df[col_name].value_counts()
    print(data)
    sns.catplot(data, ax=ax, palette=cmap)

bar_fig.savefig('images/distributions-cat.png')


data = train_df[column_names[0]].value_counts()
data_df = pd.DataFrame({'column': data.index, 'count': data.values})
sns.catplot(x=column_names[0], kind='count',
            data=train_df, ax=ax, palette=cmap)
plt.show()

#
# Categorical - distributions
#
cat_cols = [
    'ProductName',
    'Platform',
    'Processor',
    'OsPlatformSubRelease',
    'Census_DeviceFamily',
    'SkuEdition',
    'Census_PrimaryDiskTypeName',
    'Census_OSArchitecture',
    'Census_OSSkuName',
    'Census_PowerPlatformRoleName',
    'Census_MDC2FormFactor',
    'SmartScreen'
]

cat_fig = Figure(figsize=(50, 50))
cat_fig.tight_layout()
cat_nrows = 6
cat_ncols = 2
cat_axs = cat_fig.subplots(cat_nrows, cat_ncols)

for i, col_name in enumerate(cat_cols):
    r = int(i / cat_ncols)
    c = int(i % cat_ncols)
    ax = cat_axs[r, c]
    sns.countplot(
        ax=ax,
        x=col_name,
        data=train_df,
        palette='YlGnBu',
    )

    if col_name == 'Census_OSSkuName':
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=45, fontsize='xx-small')

cat_fig.savefig('images/distributions-bars.png')

percentages_df = pd.DataFrame()

for col_name in train_df.columns:
    value_counts = (train_df[col_name].value_counts(normalize=True) * 100).round(1)
    percentages = value_counts.values[:3]
    
    print(percentages_df)
    print(percentages)

    while len(percentages) < 3:
        percentages = np.append(percentages, 0.0)

    while len(percentages) > 3:
        percentages = percentages[:-1]

    percentages_df[col_name] = 0.0
    percentages_df[col_name] = percentages

percentages_df.to_csv('results/percentages.csv')

tmp_df = pd.DataFrame()
for i, col_name in enumerate(train_df.columns):
    tmp_df[col_name] = percentages_df[col_name]

    if (i + 1) % 10 == 0 or i == len(train_df.columns) - 1:
        tmp_df.to_html(f'tmp/percentages-{i}.html', index=False)
        tmp_df = pd.DataFrame()

