"""
A collection of functions covering the functionality needed to:

- Establish an experiment and track parameters, performance, etc.
- Define model parameters, architecture, and metrics thresholds.
- Train and validate a model.
- Save a trained model to disk for later use in real-time processing and scientific analysis.

"""
# imports
import datetime
import fastai
from fastai.vision.all import resnet34, \
    Adam, ImageDataLoaders, Resize, aug_transforms, cnn_learner, error_rate, accuracy, MixedPrecision, \
    ShowGraphCallback, CSVLogger, ReduceLROnPlateau, EarlyStoppingCallback, SaveModelCallback, \
    ClassificationInterpretation, load_learner
from hyperdash import Experiment as HyperdashExperiment
import logging
import matplotlib.pyplot as plt
import natsort
import numpy as np
import operator
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import sys
from tidd.metrics import confusion_matrix_scores, calculating_coverage, precision_score, recall_score, f1_score
from tidd.utils_old import Transforms, TqdmToLogger
import torch
from tqdm import tqdm
from typing import List, Union

# set logging verbosity
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # set to logging.DEBUG in development
logger = logging.getLogger()


class Model:

    """
    A class that instantiates a model for training and allows
    use of the model for inference.
    """

    def __init__(self,
                 architecture: fastai = resnet34,
                 batch_size: int = 256,
                 learning_rate: float = 0.0001,
                 optimization_function: fastai = Adam
                 ):

        self.architecture = architecture
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimization_function = optimization_function

        # some to be filled
        self.learner = None

    def fit(self, max_epochs: int, verbose: bool = False, callbacks: list = None) -> None:

        """
        Trains the specified model for a specified number of maximum epochs.
        Utilizes callbacks which save training information to CSV, reduce the
        learning rate on loss plateau, and early stop the model training process
        if performance is no longer improving when a specified list of callbacks
        is not provided by the user.
        :param max_epochs: The maximum number of epochs to train the model (default 50).
        :param verbose: If true, graphs the model during training and plots the results
        of the training process.
        :param callbacks: A list of callbacks to be used during model training. By default, includes
        CSVLogger, ReduceLROnPlateau, EarlyStoppingCallback, and SaveModelCallback.
        :return: None
        """

        # define callbacks to use during model training
        if callbacks is None:
            callbacks = [
                CSVLogger(),  # TODO: does this need a path?
                ReduceLROnPlateau(
                    monitor='valid_loss',
                    min_delta=0.1,
                    patience=2
                ),
                EarlyStoppingCallback(
                    monitor="valid_loss",
                    patience=3,
                    min_delta=0.00001
                ),
                SaveModelCallback()
            ]

        if verbose is True:
            callbacks.append(ShowGraphCallback())

        # train the model
        self.learner.fit(
            max_epochs,
            lr=self.learning_rate,
            cbs=callbacks
        )

    def export(self, export_path: Union[str, Path]) -> None:
        """
        Saves a model to local disk for later use.
        :param export_path: The location in which to save the model.
        :return: None
        """

        try:
            self.learner.export(export_path)
        except Exception as ex:
            logging.warning(RuntimeWarning, str(ex))

    def load(self, import_path: Union[str, Path]) -> None:
        """
        Loads a model from local disk for use in inference and
        other tasks.
        :param import_path: The location in which to load the model from.
        :return: None
        """

        try:
            self.learner = load_learner(import_path)
        except Exception as ex:
            logging.warning(RuntimeWarning, str(ex))

    # TODO: predict function


class Experiment:

    """
    A class that helps instantiate and manage experiments, including model training and validation.
    The Experiment class makes use of Hyperdash - please see the Hyperdash documentation for getting
    setup: https://hyperdash.io/
    """

    def __init__(self,
                 model: Model,
                 name: str = "tidd",
                 cuda_device: int = 0,
                 training_data_paths: List[Union[str, Path]] = ["./"],
                 validation_data_paths: List[Union[str, Path]] = None,
                 generate_data: bool = False,
                 share_testing: float = 0.2,
                 parallel_gpus: bool = False,
                 max_epochs: int = 50,
                 window_size: int = 60
                 ) -> None:

        # use the name to establish a Hyperdash experiment
        self.name = name
        self.model = model
        self.generate_data = generate_data
        self.exp = HyperdashExperiment(name)
        self.cuda_device = cuda_device
        self.training_data_path = training_data_paths
        self.validation_data_path = validation_data_paths
        self.share_testing = share_testing
        self.parallel_gpus = parallel_gpus
        self.max_epochs = max_epochs
        self.window_size = window_size

        # print some information
        logging.info(" ----------------------------------------------------")
        logging.info(" Experiment defined with the following parameters:")
        logging.info(" Experiment Name: " + self.name)
        logging.info(" Model Architecture: " + str(self.model.architecture.__name__))
        logging.info(" Generate data from raw files: " + str(self.generate_data))
        logging.info(" Share of Data for Testing: " + str(self.share_testing))
        logging.info(" Parallel GPUs: " + str(self.parallel_gpus))
        logging.info(" Max training epochs: " + str(self.max_epochs))
        logging.info(" Window size: " + str(self.window_size))
        logging.info(" ----------------------------------------------------\n")

        # if generate data is true, create images otherwise point to source data
        # if self.generate_data is True:




