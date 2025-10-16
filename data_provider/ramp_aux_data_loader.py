import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, data=None):
        if data is not None:
            self.fit(data)

    def fit(self, data):
        #data = np.asarray(data)
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler():
    """
    Normalize each column of the input to the range [0, 1]
    """

    def __init__(self, data=None):
        if data is not None:
            self.fit(data)

    def fit(self, data):
        #data = np.asarray(data)
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data):
        #data = np.asarray(data)
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        #data = np.asarray(data)
        return data * (self.max - self.min) + self.min



class Aux_Wind(Dataset):
    def __init__(self, args, site, flag='train', size=None,
                 features='S',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        #self.root_path = root_path
        self.site = site
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.minmax = MinMaxScaler()
        #df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                  self.data_path))

        df_raw = pd.read_csv(rf"dataset/{self.site}_labeled_dataset_aux.csv")
        columns_number = len(df_raw.columns.tolist())-1
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        #cols = list(df_raw.columns)
        #cols.remove(self.target)
        #cols.remove('date')
        #df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        print("Number of train samples: ", num_train,num_test,num_vali)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        wind_farms_number = int(columns_number/3)
        df_data_lr = df_raw.iloc[:, 1:wind_farms_number + 1]
        df_data_dr = df_raw.iloc[:, wind_farms_number + 1: 2*wind_farms_number+1] + 1.0
        df_data_diff = df_raw.iloc[:, 2*wind_farms_number+1:]



        if self.scale:
            train_data = df_raw.iloc[border1s[0]:border2s[0], 1:wind_farms_number + 1]
            self.scaler.fit(train_data.values)
            data_lr = self.scaler.transform(df_data_lr.values)
            data_dr = df_data_dr.values

            train_data_diff = df_raw.iloc[border1s[0]:border2s[0],2*wind_farms_number+1:]
            self.minmax.fit(train_data_diff.values)
            data_diff = self.minmax.transform(df_data_diff.values)

        else:
            data_lr = df_data_lr.values
            data_dr = df_data_dr.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x_dis = data_dr[border1:border2]
        self.data_x_con = data_lr[border1:border2]
        self.data_y_dis = data_dr[border1:border2]
        self.data_y_con = data_lr[border1:border2]

        self.data_diff = data_diff[border1:border2]

        self.data_stamp = data_stamp

        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        batch_x_dis = self.data_x_dis[s_begin:s_end]
        batch_x_con = self.data_x_con[s_begin:s_end]
        batch_y_dis = self.data_y_dis[r_begin:r_end]
        batch_y_con = self.data_y_con[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        batch_x_diff = self.data_diff[s_begin:s_end]
        batch_y_diff = self.data_diff[r_begin:r_end]

        return batch_x_dis, batch_x_con, batch_y_dis, batch_y_con, batch_x_diff, batch_y_diff, seq_x_mark, seq_y_mark

    def __len__(self):
        # print("len(self.data_x_dis) - self.seq_len - self.pred_len + 1",len(self.data_x_dis) - self.seq_len - self.pred_len + 1)
        return len(self.data_x_dis) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

