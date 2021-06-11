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
    Adam, ImageDataLoaders, Resize, aug_transforms, cnn_learner, error_rate, accuracy, \
    ShowGraphCallback, CSVLogger, ReduceLROnPlateau, EarlyStoppingCallback, SaveModelCallback, \
    ClassificationInterpretation, load_learner
from hyperdash import Experiment as HyperdashExperiment
import json
import logging
import matplotlib.pyplot as plt
import natsort
import numpy as np
import operator
import os
from pathlib import Path
import sys
from tidd.metrics import confusion_matrix_scores, calculating_coverage, \
    precision_score, recall_score, f1_score, confusion_matrix_classification
from tidd.plotting import plot_classification, plot_distribution
from tidd.utils import Data, TqdmToLogger, Transform
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

    def predict(self, test_item: Union[str, Path, np.ndarray, torch.Tensor]) -> tuple:
        """
        wrapper function around the fastai.Learner.predict function

        :param test_item: either the file path or the tensor/array object
                            single item for the predictor
        :return: a tuple containing the predicted label and the confidence values for the classes.
        """
        prediction = self.learner.predict(test_item)
        return prediction[0], np.max(prediction[2].cpu().detach().numpy())

    def predict_sequences(self, test_items_list: list) -> tuple:
        """
        returns the list of prediction labels and confidence values
        wrapper for the Model.predict over multiple items

        :param test_items_list: list of Union[str, Path, np.ndarray, torch.Tensor]
                                to be predicted over
        :return: list of class labels, list of np.ndarrays of confidence values
                list of ints 1 is an anomalous index

        """
        classifications = list()
        confidence_values = list()

        for item in test_items_list:

            label, confidence = self.predict(item)
            classifications.append(label)
            confidence_values.append(confidence)

        # generate boolean classification
        classification_bool = [0 if x == "normal" else 1 for x in classifications]
        
        return classifications, confidence_values, classification_bool


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
                 window_size: int = 60,
                 save_path: Union[str, Path] = "./"
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
        
        # initialize some attributes / objects for validation (out of sample)
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tp_lengths = list()
        self.fp_lengths = list()

        # initialize an object in which to store metrics
        self.metrics = dict()

        # create the save_path dir if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.save_path = save_path

        # print some information
        logging.info(" ----------------------------------------------------")
        logging.info(" Experiment defined with the following parameters:")
        logging.info(" Experiment Name: " + self.name)
        logging.info(" Model Architecture: " + str(self.model.architecture.__name__))
        logging.info(" Generate data from raw files: " + str(self.generate_data))
        logging.info(" Share of data for testing: " + str(self.share_testing))
        logging.info(" Parallel GPUs: " + str(self.parallel_gpus))
        logging.info(" Max training epochs: " + str(self.max_epochs))
        logging.info(" Window size: " + str(self.window_size))
        logging.info(" ----------------------------------------------------\n")

        # if generate data is true, create images otherwise point to source data
        if self.generate_data is True:

            logging.info("Generating image dataset for experiment...")

            self.training_data_path, self.validation_data_path = Data.prepare_training_validation_data(
                experiment_name=self.name,
                training_data_paths=training_data_paths,
                validation_data_paths=validation_data_paths,
                window_size=window_size
            )

            # TODO: after creation assign the path to the training and validation data

        else: 
            # TODO: point to the source data
            pass

        # prep the Experiment object
        logging.info("Specifying CUDA device...")
        self._set_cuda()

        logging.info("Loading data from disk...")
        self._set_data()

        logging.info("Setting model parameters...")
        self._set_model()

        # experiment object ready
        logging.info("Experiment ready.")

    def _set_cuda(self) -> None:
        """
        Checks that CUDA is available for model training and that the specified device is available. Returns a
        warning to the user when this is not the case.
        :return: None
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
                self.model.learner = torch.nn.DataParallel(self.model.learner)

            else:
                # emit a UserWarning
                logging.warning(UserWarning, "Only 1 CUDA device available.")
                self.exp.param("parallel_gpus", False)

        self.exp.param("parallel_gpus", self.parallel_gpus)

    def _set_data(self, verbose: bool = False) -> None:
        """
        Prepares the data for the experiment based on the Experiment
        and Model parameters.
        :param verbose: When set to True, plots a small randomly-sampled
        batch of the data loaded into the experiment.
        :return: None
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
        """
        Prepares the model for the experiment based on the Experiment
        and Model parameters.
        :return: None
        """

        save_path = self.save_path + '/' + 'model_output'
        Path(save_path).mkdir(parents=True, exist_ok=True)

        self.model.learner = cnn_learner(
            self.dls,  # data
            self.model.architecture,  # architecture
            metrics=[error_rate, accuracy],  # metrics
            path= save_path,
            pretrained=False,  # whether or not to use transfer learning
            normalize=True,  # this function adds a Normalization transform to the dls
            opt_func=self.model.optimization_function  # SGD # optimizer
        )

        # add the model parameters to the Hyperdash experiment
        self.exp.param("batch_size", self.model.batch_size)
        self.exp.param("architecture", self.model.architecture)
        self.exp.param("learning_rate", self.model.learning_rate)
        self.exp.param("epochs_max", self.max_epochs)

    def _out_of_sample(self, verbose: bool = False) -> None:

        """
        Using data in the specified validation_paths and raw data prior to image conversion, performs an
        out of sample validation using the trained model and a set of ground truth labels. Records out of
        sample validation metrics in the metrics attribute.
        :param verbose: Default false. When true, plots progress bars and saves plots in the save_path.
        :return: None
        """

        # define the save path for the out of sample output and make sure the path exists
        save_path = self.save_path + '/' + 'out_of_sample'
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # subdue the progress bar as to not clog the stdout / cell output
        with self.model.learner.no_bar():
            # disable logging
            self.model.learner.no_logging()

            # initialize some metrics counting
            self.tp = 0
            self.fn = 0 
            self.fp = 0 
            self.tp_lengths = list()
            self.fp_lengths = list()

            # get the full path of each directory containing image files
            image_directories = Data._get_image_directories(self.validation_data_path)
            # filter for those containing "unlabeled"
            image_directories = [i for i in image_directories if "unlabeled" in i]

            # read in labels 
            labels_path = self.validation_data_path.split(self.name)[0] + "/" + self.name + "/tid_start_finish_times.json"
            with open(labels_path, "rb") as f_in:
                labels = json.load(f_in)
                locations = list(labels.keys())
            tqdm_out = TqdmToLogger(logger, level=logging.INFO)
            
            # TODO (future work): parallel process the below
            # process each of the directories in the validation set
            for d in tqdm(image_directories, file=tqdm_out, total=len(image_directories), mininterval=10, disable=operator.not_(verbose)):

                try:

                    # get the satellite amd ground station from the path
                    ground_station = d.split("__")[0].split("/")[-1]
                    sat = d.split("__")[1].split("/")[0]
                    pass_id = ground_station + "__" + sat

                    # get the images in the directory
                    image_files = [d + "/" + f for f in
                            natsort.natsorted(os.listdir(d)) if ".jpg" in f and
                            f[0] != "."]

                    # determine the location
                    location = None
                    for l in locations:
                        if l in d:
                            location = l
                            break

                    # assign the doy for the location
                    # note: assumes one doy for the location
                    doy_for_location = list(labels[location].keys())[0]

                    # filter for images that match the day of year
                    image_files_with_labels = list()
                    for i in image_files:
                        doy = i.split("/")[-1].split("_")[0]
                        if doy in labels[location].keys():
                            image_files_with_labels.append(i)

                    # for the sorted images in the directory, predict the sequence
                    try:
                        classification, classification_confidence, classification_bool = self.model.predict_sequences(
                            image_files_with_labels)
                    except Exception as ex:
                        logging.warning(RuntimeWarning, "Error encountered when predicting sequence. Not included in "
                                                        "validation result.")
                        logging.warning(str(ex))
                        if ex is KeyboardInterrupt:
                            break
                        continue

                    # get the indices of the anomalous values
                    anom_idx = np.where(np.array(classification_bool) == 1)
                    # get the sequences of the anomalous values
                    anom_sequences = Transform.group_consecutive_values(list(anom_idx[0]))

                    # we need to load in the original data file (float data) that contains the second of day
                    # using the doy and sat of first image load in the data file
                    file_path = None
                    location_path = d.split("experiments")[0]
                    file_paths = list()
                    for path, subdirs, files in os.walk(location_path + location):
                        for name in files:
                            file_paths.append(os.path.join(path, name))

                    # cycle through the file paths until we identify the matching path
                    # note: assumes only one path matches (which should be the case but is not checked here)
                    for i in file_paths:
                        if doy_for_location in i and location in i and sat in i and ground_station in i:
                            file_path = i

                    # read the original data file (pre-image)
                    df = Data.read_data_from_file(file_path)

                    # todo: refactor below so that this switch is not hard-coded
                    year = 2012
                    if location == "chile":
                        year = 2015

                    # convert sod to timestamp
                    df = Transform.sod_to_timestamp(
                        df,
                        year=year,
                        day_of_year=int(doy_for_location)
                    )

                    # resample to 1 min
                    # TODO: expose as param
                    df = df.resample("1min").mean()

                    # transform values by first getting the individual events
                    events = Transform().split_by_nan(
                        dataframe=df,
                        min_sequence_length=100
                    )

                    # get the ground truth for the out of sample assessment
                    # note: currently assumes one ground truth sequence in period
                    event = events[0].reset_index()
                    ground_truth_sequence = event[
                        (event["sod"] >= labels[location][doy_for_location][sat]["start"]) &
                        (event["sod"] >= labels[location][doy_for_location][sat]["finish"])
                    ].index.values

                    # adjust the sequence for the window size used to generate the images
                    adjusted_ground_truth_sequence = [x - self.window_size for x in ground_truth_sequence]

                    # Obtain the true positives, false negatives, and false positives
                    tp, fn, fp, tp_lengths, fp_lengths = confusion_matrix_classification(
                        adjusted_ground_truth_sequence, # note: currently assumes one ground truth sequence in period
                        anom_sequences
                    )

                    self.tp += tp
                    self.fn += fn
                    self.fp += fp
                    [self.tp_lengths.append(tpl) for tpl in tp_lengths]
                    [self.fp_lengths.append(fpl) for fpl in fp_lengths]

                    # if verbose, save plots from validation
                    if verbose:

                        fig = plot_classification(
                            event,
                            pass_id,
                            labels[location][str(doy_for_location)][sat],
                            classification_bool,
                            classification_confidence,
                            window_size_adjustment=self.window_size
                        )
                        fig.savefig(save_path + '/' + pass_id + '_classification_plot.png')
                        plt.close(fig)

                    if self.save_path is not None:
                        try:
                            save_df = event.iloc[59:].copy()
                            save_df['anomaly'] = classification_bool
                            save_df['confidence'] = classification_confidence
                            save_df.to_csv(save_path + "/" + pass_id + '_results.csv')
                        except FileNotFoundError:
                            logging.info('Save Path "' + save_path + "/" + pass_id + '_results.csv' + '" not valid')

                except Exception as ex:
                    if ex is KeyboardInterrupt:
                        break
                    else:
                        logging.warning(RuntimeWarning, "Error encountered when running out of sample validation for "
                                                        "sequence. Not included in validation result.")
                        logging.warning(str(ex))
                        continue

            # calculate validation metrics
            precision = precision_score(self.tp, self.fp)
            recall = recall_score(self.tp, self.fn)
            f_score = f1_score(precision, recall)

            # record in Hyperdash
            self.metrics["validation_precision"] = precision
            precision = self.exp.metric("validation_precision", precision)
            recall = self.exp.metric("validation_recall", recall)
            self.metrics["validation_recall"] = precision
            f_score = self.exp.metric("validation_f1_score", f_score)
            self.metrics["validation_f1_score"] = precision

            # get the mean sequence lengths and record in Hyperdash
            tp_sequence_length = np.mean(self.tp_lengths)
            self.metrics["tp_sequence_length"] = tp_sequence_length
            tp_sequence_length = self.exp.metric("tp_sequence_length", tp_sequence_length)
            fp_sequence_length = np.mean(self.fp_lengths)
            self.metrics["fp_sequence_length"] = fp_sequence_length
            fp_sequence_length = self.exp.metric("fp_sequence_length", fp_sequence_length)

            # if verbose, plot the distribution of the sequence lengths
            if verbose is True:
                ax = plot_distribution(self.tp_lengths, self.fp_lengths)
                ax.savefig(save_path + "/classification_sequence_length_distribution.jpg")

            logging.info("Out of sample validation complete.")

    def run(self, verbose: bool = False) -> None:

        """
        Runs the Experiment according to the specified Experiment and Model
        parameters.
        :param verbose: If True, shows Model training information and loss curves and plots
        results visually for interpretation.
        :return: None
        """

        # train the model
        self.model.fit(max_epochs=self.max_epochs, verbose=verbose)

        # interpret the results
        interp = ClassificationInterpretation.from_learner(self.model.learner)

        # calculate the confusion matrix and get the scores
        cm = interp.confusion_matrix()
        results = confusion_matrix_scores(cm)

        # track results in the Hyperdash experiment
        self.exp.metric("training_accuracy", results[0])
        self.metrics["training_accuracy"] = results[0]
        self.exp.metric("training_precision", results[1])
        self.metrics["training_precision"] = results[1]
        self.exp.metric("recall", results[2])
        self.metrics["training_recall"] = results[2]
        self.exp.metric("f1_score", results[3])
        self.metrics["training_f1_score"] = results[3]

        # calculate the coverage
        predictions, targets = self.model.learner.get_preds()  # by default uses validation set
        anom_cov, normal_cov = calculating_coverage(predictions, targets)
        self.exp.metric("anomaly coverage", anom_cov)
        self.metrics["training_anomaly_coverage"] = anom_cov
        self.exp.metric("normal coverage", normal_cov)
        self.metrics["training_normal_coverage"] = normal_cov

        # if verbose, show results
        if verbose is True:

            # plot the results
            interp.plot_top_losses(9, figsize=(15, 11))
            interp.plot_confusion_matrix(figsize=(4, 4), dpi=120)

        # as part of the Experiment, perform an out-of-sample (OOS) validation of the results
        self._out_of_sample(verbose=verbose)

        # end the experiment
        self.exp.end()





