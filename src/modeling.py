from fastai.tabular import *
from fastai.metrics import *
from fastai import torch_core
from fastai.callbacks import *
from fastai.callbacks.mem import PeakMemMetric
from fastai.utils.mod_display import *

import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from PyNomaly import loop
import random
from sklearn.preprocessing import minmax_scale
import seaborn as sns
sns.set(style="darkgrid")
from scipy import spatial, stats
from src import data
import time
import torch
from typing import Union
from tqdm.notebook import tqdm
import csv
import warnings


class Model:
    
    ## Work in progress
    @staticmethod
    def make_data_bunch(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame, 
        df_valid: pd.DataFrame, 
        features: list, 
        dependent: str, 
        cuda_device: torch.device, #added cuda device here 
        include_catvars: bool = False, 
        catvars: list = [], 
        batch_size: int = 256
    ) -> dict:
        """
        Creates a TabularDataBunch to feed as an input 
        into the learner. 
        """

        valid_start_index = df_train.shape[0] + 1
        valid_end_index = df_train.shape[0] + df_test.shape[0]

        df_train_validation = pd.concat([df_train, df_test])

        # create the data bunch
        if include_catvars:
            data = TabularDataBunch.from_df(
                "models", 
                df_train_validation[features + [dependent]], 
                dependent, 
                valid_idx=np.array(list(range(valid_start_index, valid_end_index))),
    #             test_df=df_test, 
                procs=[Categorify],
                bs=batch_size, # batch size
                cat_names=catvars,
                device=cuda_device,
                num_workers=0
            )
        else:
            data = TabularDataBunch.from_df(
                "models", 
                df_train_validation[features + [dependent]], 
                dependent, 
                valid_idx=np.array(list(range(valid_start_index, valid_end_index))),
    #             test_df=df_test, 
                procs=None, # disable any automatic preprocessing
                bs=batch_size, # batch size
                device=cuda_device,
                num_workers=0
            )

        return {
            "databunch": data, 
            "train": df_train, 
            "test": df_test, 
            "valid": df_valid
        }
    
    @staticmethod
    def find_appropriate_lr(model: Learner, lr_diff:int = 15, loss_threshold:float = .001, adjust_value:float = 1, plot:bool = False) -> float:
        
        """
        We utilize an [automatic learning rate finder](https://forums.fast.ai/t/automated-learning-rate-suggester/44199/8) to 
        determine the ideal learning rate automatically. While this approach does not always guarantee that the perfect 
        learning rate is found, in practice we have found the approach to work well and has been quite stable.
        
        adjusted the loss threshold to 0.001 from 0.05
        """
        
        #Run the Learning Rate Finder
        model.lr_find(
            end_lr=2.,
            stop_div=False # continues through all LRs as opposed to auto stopping
        )

        #Get loss values and their corresponding gradients, and get lr values
        losses = np.array(model.recorder.losses)
        assert(lr_diff < len(losses))
        loss_grad = np.gradient(losses)
        lrs = model.recorder.lrs

        #Search for index in gradients where loss is lowest before the loss spike
        #Initialize right and left idx using the lr_diff as a spacing unit
        #Set the local min lr as -1 to signify if threshold is too low
        local_min_lr = -1
        r_idx = -1
        l_idx = r_idx - lr_diff
        while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
            local_min_lr = lrs[l_idx]
            r_idx -= 1
            l_idx -= 1

        lr_to_use = local_min_lr * adjust_value
        

        if plot:
            # plots the gradients of the losses in respect to the learning rate change
            plt.plot(loss_grad)
            plt.plot(len(losses)+l_idx, loss_grad[l_idx], markersize=10, marker='o', color='red')
            plt.ylabel("Loss")
            plt.xlabel("Index of LRs")
            plt.show()

            plt.plot(np.log10(lrs), losses)
            plt.ylabel("Loss")
            plt.xlabel("Log 10 Transform of Learning Rate")
            loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
            plt.plot(np.log10(lr_to_use), loss_coord, markersize=10,marker='o',color='red')
            plt.show()
        
        
#         return 0.001
        return lr_to_use
    
    @staticmethod
    def predict_values(dataframe: pd.DataFrame, learner: tabular_learner, dependent: str, frac: float = 1.0) -> pd.DataFrame:
    
        """
        Using the passed learner, predicts the appropriate value given the input data 
        and generates errors for analysis. 
        
        Optional frac parameter allows user to specify whether they would like to predict 
        over the entirety of the dataframe or over a smaller subset of data. 
        """

        # get a sample of the dataset 
        dataframe_pred = dataframe.copy()
        n_obs = int(dataframe_pred.shape[0] * frac)
        idx = random.sample(range(0, dataframe_pred.shape[0]), n_obs)
        dataframe_pred = dataframe_pred.iloc[idx, :]

        # get the predictions
        predictions = []
        print('Generating predictions and errors...')
        for idx, row in tqdm(dataframe_pred.iterrows(), total=dataframe_pred.shape[0]):        
            predictions.append(learner.predict(row)[1].numpy()[0])


        dataframe_pred["predicted"] = predictions
        dataframe_pred['error'] = dataframe_pred['predicted'] - dataframe_pred[dependent]
        dataframe_pred['absolute_error'] = np.abs(dataframe_pred['error'])
        dataframe_pred["timestamp"] = dataframe_pred.index

        return dataframe_pred
    
    # TODO: def classify 
    # this would use our error (residual) handling approach of choice to classify anomalies 
    # later assessed by the metrics class, which is applied over multiple classifications 

class Data:
    
    @staticmethod
    def get_periods(dataframe: pd.DataFrame) -> list: 
        
        """
        These periods are defined by consecutive empty (NaN) values in the dataframe. 
        The data is only available for the satellite as it passes close to the ground 
        station on each day. We could train the data in a similar way, ensuring our 
        approach is compatible with the constraints in the operating environment. 
        """

        # handle missing values and "chunk" the data for training and testing 
        events = np.split(dataframe, np.where(np.isnan(dataframe))[0])

        # removing NaN entries
        events = [ev[~np.isnan(ev)] for ev in events if not isinstance(ev, np.ndarray)]

        # removing empty DataFrames
        events = [ev.dropna() for ev in events if not ev.empty and ev.shape[0] > 100]

        return events
    
    @staticmethod
    def rescale_periods(events: list, feature_range: tuple = (-1, 1)) -> list: 
        """
        Rescales the values according to the specified feature range, default (-1, 1). 
        Reach period (typically a single day) is rescaled independently, mimicking the 
        proposed implementation of the model in practice (where it would be retrained 
        each day during the off-line period for the satellite and ground station). 
        Returns a list of normalized data, one entry for each period. 
        """
    
        normalized_events = list()
        for ev in events: 

            # for each column in the data, rescale -1 to 1 
            col_data = list()
            for col in ev.columns.values:

                normalized_data = minmax_scale(
                            ev[col].dropna(), 
                            feature_range=feature_range
                        )
                col_data.append(normalized_data)

            df_period = pd.DataFrame(np.array(col_data).T, columns=list(ev.columns.values) )
            df_period["timestamp"] = ev[col].index
            df_period.index = df_period["timestamp"]
            df_period = df_period.drop(columns=["timestamp"])

            # convert to seconds of the day for later annotation 
            df_period["sod"] = (df_period.index.hour*60+df_period.index.minute)*60 + df_period.index.second

            normalized_events.append(df_period)
            
        return normalized_events
    
    @staticmethod
    def allocate_periods(periods: list, index_earthquake: int) -> dict:
        """
        Based on which period is specified as containing the anomaly to be 
        detected, creates a training, testing, and validation set that can 
        be used to assess the approach and later obtain quantitative 
        performance metrics. The training set contains data that is pre-anomaly,
        with model training being guided by the single-period test set (the period 
        before the period with the anomaly). The validation set contains the period 
        of the anomaly. Any periods after the anomaly period are currenly not used 
        as part of the experimental process. 
        """
    
        data = dict()

        data["valid"] = periods[index_earthquake]
        data["test"] = periods[index_earthquake - 1]
        data["train"] = periods[0:index_earthquake - 1]

        return data