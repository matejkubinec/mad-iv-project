import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from matplotlib.figure import Figure
from src.load_data import load_data

columns_to_drop = [
    'ProductName',
    'IsBeta',
    'RtpStateBitfield',
    'IsSxsPassiveMode',
    'AVProductsEnabled',
    'HasTpm',
    'Platform',
    'OsVer',
    'AutoSampleOptIn',
    'PuaMode',
    'SMode',
    'Firewall',
    'UacLuaenable',
    'DeviceFamily',
    'IsPortableOperatingSystem',
    'IsFlightingInternal',
    'IsFlightsDisabled',
    'ThresholdOptIn',
    'IsWIMBootEnabled',
    'IsVirtualDevice',
    'IsPenCapable',
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