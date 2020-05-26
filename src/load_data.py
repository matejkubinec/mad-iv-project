import pandas as pd


def load_data():
    train_df = pd.read_csv('data/train.csv', index_col=0, header=0)

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

    for col in cols['version']:
        codes, uniques = pd.factorize(train_df[col])
        train_df[col] = codes

    for col in cols['categorical']:
        codes, uniques = pd.factorize(train_df[col])
        train_df[col] = codes

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

    return train_df.drop(columns=cols_to_drop)
