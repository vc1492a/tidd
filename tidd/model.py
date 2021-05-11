"""
A collection of functions covering the functionality needed to:

- Establish an experiment and track parameters, performance, etc.
- Define model parameters, architecture, and metrics thresholds.
- Train and validate a model.
- Save a trained model to disk for later use in real-time processing and scientific analysis.

"""

# imports
import fastai
from fastai.vision.all import resnet34, \
    Adam, ImageDataLoaders, Resize, aug_transforms, cnn_learner, error_rate, accuracy, MixedPrecision, \
    ShowGraphCallback, CSVLogger, ReduceLROnPlateau, EarlyStoppingCallback, SaveModelCallback, \
    ClassificationInterpretation, load_learner
from hyperdash import Experiment as HyperdashExperiment
import logging
from pathlib import Path
import torch
from typing import Union
from tidd.metrics import confusion_matrix_scores, calculating_coverage


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
        :callbacks param: A list of callbacks to be used during model training. By default, includes
        CSVLogger, ReduceLROnPlateau, EarlyStoppingCallback, and SaveModelCallback.
        :return: None
        """

        # define callbacks to use during model training
        if callbacks is None:
            callbacks = [
                CSVLogger(),  # TODO: does this need a path?
                #         ParamScheduler(sched),
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

        # TODO: docstring

        # TODO: add exception handlers, checks for success
        self.learner.export(export_path)

    def load(self, import_path: Union[str, Path]) -> None:

        # TODO: docstring

        # TODO: add exception handlers, checks for success
        self.learner = load_learner(import_path)


class Experiment:

    """
    A class that helps instantiate and manage experiments, including model training and validation.
    The Experiment class makes use of Hyperdash - please see the Hyperdash documentation for getting
    setup: https://hyperdash.io/
    """

    # TODO: add typing for model
    def __init__(self,
                 model: Model,
                 name: str = "tidd",
                 cuda_device: int = torch.cuda.current_device(),
                 training_data_path: Union[str, Path] = "./",
                 validation_data_path: Union[str, Path] = "./",
                 test_percent: float = 0.2,
                 parallel_gpus: bool = False,
                 max_epochs: int = 50,
                 coverage_threshold: float = 0.9 # TODO: add check to restrict to a number between [0, 1]
                 ) -> None:

        # TODO: add docstring

        # use the name to establish a Hyperdash experiment
        self.model = model
        self.exp = HyperdashExperiment(name)
        self.cuda_device = cuda_device
        self.training_data_path = training_data_path
        self.validation_data_path = validation_data_path
        self.test_percent = test_percent
        self.parallel_gpus = parallel_gpus
        self.max_epochs = max_epochs
        self.coverage_threshold = coverage_threshold

        # some to be filled
        self.dls = None

        # prep the Experiment object
        self._set_cuda()
        self._set_data()
        self._set_model()

    def _set_cuda(self) -> None:
        """
        Checks that CUDA is available for model training and that the specified device is available. Returns a
        warning to the user when this is not the case.
        """

        try:
            assert torch.cuda.is_available()
            torch.cuda.set_device('cuda:' + str(self.cuda_device))
            self.exp.param("device_name", torch.cuda.get_device_name(self.cuda_device))
        except AssertionError:
            logging.warning("Specified CUDA device not available. No device_name experiment parameter sent.")

        # TODO:
        if self.parallel_gpus is True:

            if torch.cuda.device_count() > 1:
                # TODO: the below may need to be the fastai object
                self.model.model = torch.nn.DataParallel(self.model.model)

            else:
                # emit a UserWarning
                logging.warning(UserWarning, "Only 1 CUDA device available.")
                self.exp.param("parallel_gpus", False)

        self.exp.param("parallel_gpus", self.parallel_gpus)

    def _set_data(self, verbose: bool = False) -> None:
        """
        Prepares the data for the experiment based on the Experiment and Model parameters.
        """

        self.dls = ImageDataLoaders.from_folder(
            self.training_data_path,
            item_tfms=Resize(224),
            valid_pct=0.2,
            bs=self.model.batch_size,
            ds_tfms=aug_transforms(do_flip=True, flip_vert=True)
        )

        # set the data attributes in the Hyperdash experiment
        self.exp.param("data_path_train", self.training_data_path)

        if verbose is True:
            self.dls.show_batch()

    def _set_model(self):

        self.model.learner = cnn_learner(
            self.dls,  # data
            self.model.architecture,  # architecture
            metrics=[error_rate, accuracy],  # metrics
            pretrained=False,  # whether or not to use transfer learning
            normalize=True,  # this function adds a Normalization transform to the dls
            #     callback_fns=[]
            opt_func=Adam  # SGD # optimizer
        )

        # add the model parameters to the Hyperdash experiment
        self.exp.param("batch_size", self.model.batch_size)
        self.exp.param("architecture", self.model.architecture)
        self.exp.param("learning_rate", self.model.learning_rate)
        self.exp.param("epochs_max", self.max_epochs)

    # def _out_of_sample(self):



    def run(self, verbose: bool = False) -> None:

        # TODO: docstring

        # train the model
        self.model.fit(max_epochs=self.max_epochs, verbose=verbose)

        # interpret the results
        interp = ClassificationInterpretation.from_learner(self.model.learner)

        cm = interp.confusion_matrix()

        results = confusion_matrix_scores(cm)

        # track results in the Hyperdash experiment
        self.exp.metric("accuracy", results[0])
        self.exp.metric("precision", results[1])
        self.exp.metric("recall", results[2])
        self.exp.metric("F1 Score", results[3])

        # calculate the coverage
        predictions, targets = self.model.learner.get_preds()  # by default uses validation set
        anom_cov, normal_cov = calculating_coverage(predictions, targets, self.coverage_threshold)
        self.exp.metric("anomaly coverage", anom_cov)
        self.exp.metric("normal coverage", normal_cov)

        # if verbose, show results
        if verbose is True:

            # plot the results
            interp.plot_top_losses(9, figsize=(15, 11))
            interp.plot_confusion_matrix(figsize=(4, 4), dpi=120)

        # TODO: as part of the Experiment, perform an out-of-sample (OOS) validation of the results!

        # end the experiment
        self.exp.end()




