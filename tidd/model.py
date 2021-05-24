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
from tidd.utils import Transforms, TqdmToLogger
import torch
from tqdm import tqdm
from typing import Union

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


class Experiment:

    """
    A class that helps instantiate and manage experiments, including model training and validation.
    The Experiment class makes use of Hyperdash - please see the Hyperdash documentation for getting
    setup: https://hyperdash.io/
    """

    def __init__(self,
                 model: Model,
                 name: str = "tidd",
                 cuda_device: int = torch.cuda.current_device(),
                 training_data_paths: Union[str, Path] = "./",
                 validation_data_paths: Union[str, Path] = "./",
                 test_percent: float = 0.2,
                 parallel_gpus: bool = False,
                 max_epochs: int = 50,
                 coverage_threshold: float = 0.9, # TODO: add check to restrict to a number between [0, 1]
                 window_size: int = 60
                 ) -> None:

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
        self.window_size = window_size

        # some to be filled
        self.dls = None
        self.tp_lengths = list()
        self.fp_lengths = list()
        self.tp = 0
        self.fp = 0
        self.fn = 0

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

        self.model.learner = cnn_learner(
            self.dls,  # data
            self.model.architecture,  # architecture
            metrics=[error_rate, accuracy],  # metrics
            pretrained=False,  # whether or not to use transfer learning
            normalize=True,  # this function adds a Normalization transform to the dls
            opt_func=self.model.optimization_function  # SGD # optimizer,
            # model_dir="" # TODO
        )

        # add the model parameters to the Hyperdash experiment
        self.exp.param("batch_size", self.model.batch_size)
        self.exp.param("architecture", self.model.architecture)
        self.exp.param("learning_rate", self.model.learning_rate)
        self.exp.param("epochs_max", self.max_epochs)

    # TODO: consider moving to Data
    @staticmethod
    def _get_directories(dir_path: Union[str, Path]) -> list:
        """
        # TODO
        """

        # get the directories at the specified path
        directories = [
            d for d in os.listdir(dir_path) if os.path.isdir(dir_path + '/' + d)
        ]

        return directories

    # TODO: consider moving to Data
    @staticmethod
    def _get_image_files(dir_path: Union[str, Path], subdir_path: Union[str, Path]) -> list:

        """
        # TODO
        """

        # get all of the image paths in that directory for the day of the earthquake
        base_path = dir_path + "/" + subdir_path + "/unlabeled"
        image_files = [base_path + "/" + f for f in natsort.natsorted(os.listdir(base_path)) if
                       ".jpg" in f and "302" in f.split("_")[0]]

        return image_files

    # TODO: align docstring with rest of code base
    @staticmethod
    def _period_class(length_images: int, windows: list, classification: list,
                      classification_confidence: list) -> pd.DataFrame:
        """
        abstracted logic to classify anomalies with different weighting schemes,
        currently we utilize a naive approach towards classification

        :input length_images: an integer representing the amount of images
        :input windows: a list of 2-element lists where each 2-element list contains
        the start index and end index of an anomalous sequence
        :input classification: list of the classification at each index
        :input classification confidence: the confidence value of the classification at the index
        :output: datafram with the classification confidence
        """
        window_end = windows[-1][1] + 1

        period_classification_df = pd.DataFrame(
            index=list(range(0, length_images)),
            columns=list(range(0, window_end - 1))
        ).astype(float)

        # note that this matrix can be used to classify anomalies with different weighting schemes
        # this is left for future work. We take the naive approach and say that any time index
        # identified as anomaly is one
        # and fill
        index = 0
        for c, cc, w in zip(classification, classification_confidence, windows):
            if c == "normal":
                val = 0
            else:
                val = 1
            period_classification_df.iloc[index, w[0]:w[1]] = val
            index += 1

        return period_classification_df

    # TODO: refactor the below completely. Decide how to load in data once for one experiment
    # TODO (cont): maybe images are generated as part of an experiment using the provided ground truth, and are
    # TODO (cont): then used. That way, raw data is only read in once in the Exp object instead of being read in
    # TODO (cont): again just for validation.

    @staticmethod
    def _read_data(ground_station_name: str, sat_name: str) -> pd.DataFrame:
        """
        simple read data function that converts the txt data files into pandas dataframes
        :input ground_station_name:
        :input sat_name:
        :output: pd.DataFrame
        """
        pass_id = ground_station_name + "__" + sat_name
        try:
            # TODO: below is hard-coded - need to make dynamic
            sat = "../data/hawaii/2012/302/" + ground_station_name + "3020.12o_" + sat_name + ".txt"

            f = open(sat, 'r')
            line1 = f.readline()

            line1 = line1.replace('#', '').replace("dsTEC/dt [TECU/s]", "dsTEC/dt").replace("elev", "ele")
            rename_cols = line1.split()
            rename_cols.remove("sod")
            new_cols = list()

            # rename the columns
            for rn_col in rename_cols:
                new_col = pass_id + "_" + rn_col
                if rn_col == "dsTEC/dt":
                    new_col = pass_id
                new_cols.append(new_col)
            new_cols = ["sod"] + new_cols

            df = pd.read_table(
                sat,
                index_col='sod',
                sep="\t\t| ",
                names=new_cols,
                engine="python",
                skiprows=1
            )

            new_cols.remove('sod')

            sod = df.index
            timestamps = list()
            date = datetime.datetime(2012, 1, 1) + datetime.timedelta(302 - 1)

            for s in sod:
                # hours, minutes, seconds
                hours = int(s // 3600)
                minutes = int((s % 3600) // 60)
                seconds = int((s % 60))

                # create a datetime object and append to the list
                date_time = datetime.datetime(date.year, date.month, date.day, hours, minutes, seconds)
                timestamps.append(date_time)

            df["timestamp"] = timestamps
            new_cols.append("timestamp")

            # now that we have read in the data, do some formatting
            df = df[new_cols].reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.index = df["timestamp"]
            df = df.resample("1Min").mean()

            return df

        except Exception as ex:
            print(RuntimeWarning, str(ex))

    def _confusion_matrix_classification(self, adjusted_ground_truth_sequence: list,
                                         anom_sequences: list) -> tuple:
        """
        Records the total count of true positives, false negatives, and false positives

        A true positive is recorded if any portion of a predicted sequence of anomalies falls within any
        true labeled sequence. Only one true positive is recorded
        even if portions of multiple predicted sequences fall within
        a labeled sequence.

        If no predicted sequences overlap with a positively labeled
        sequence, a false negative is recorded for the labeled sequence

        For all predicted sequences that do not overlap a labeled
        anomalous region, a false positive is recorded

        Also modifies tp_lengths and fp_lengths


        :input adjusted_ground_truth_sequences: list of labels for each minute
        :input anom_sequences: list of indices where the minute sequences are considered anomalous
        :input tp_lengths: list of the length of the true anomalous seq
        :input fp_lengths: list of the lengths of the false anomalous seq
        :output: tuple of 3 integers (true positive, false negative, false positive)
        """

        tp = 0
        fn = 0
        fp = 0

        # NOTE: current code assumes one anomalous sequence in ground truth

        # check for false positives and true positives
        for anom_seq in anom_sequences:

            intersection = list(set(adjusted_ground_truth_sequence) & set(anom_seq))

            if len(intersection) > 0:
                tp = 1
                fn = 0

                self.tp_lengths.append(len(anom_seq))

            else:
                fp += 1

                self.fp_lengths.append(len(anom_seq))

        # check for false negatives
        fn = 1
        for anom_seq in anom_sequences:
            intersection = list(set(adjusted_ground_truth_sequence) & set(anom_seq))
            if len(intersection) > 0:
                fn = 0
                break

        return tp, fn, fp

    # TODO: move to plotting class / file
    @staticmethod
    def _plot_distribution(tp_lengths: list, fp_lengths: list, save_path: Union[str, Path] = "./output") -> None:
        """
        creates the plot of visualizing the distribution of true and false positives
        :input: list of true positive sequences
        :input: list of false positive sequences

        # TODO: update docstring

        """
        # first organize the data into a dataframe
        rows = list()

        for i in tp_lengths:
            rows.append([i, "true_positive"])
        for j in fp_lengths:
            rows.append([j, "false_positive"])

        df_sequence_lengths = pd.DataFrame(
            rows,
            columns=["sequence_length", "sequence_type"]
        )

        colors = ["#4A91C2", "#16068A"]
        # Set your custom color palette
        ax = sns.displot(df_sequence_lengths, x="sequence_length", hue="sequence_type", kde=True, multiple="stack",
                         palette=sns.color_palette(colors))

        plt.savefig(save_path + "/classification_sequence_length_distribution.png")

    # TODO: move to plotting class / file
    @staticmethod
    def _plot_classification(event: pd.DataFrame,
                             pass_id: str,
                             sat_annotation: dict,
                             classification_bool: list,
                             classification_confidence: list,
                             save_path: Union[str, Path] = "./output"
                             ) -> None:
        """
        Creates 3 plots
        Plot of the time series data with markings of the start and end of the anomalous sequence
        Plot of the predictions for each sequence 1 hour
        Plot of the confidence levels for each of the predictions

        # TODO: update docstring

        :input event: DataFrame containing the float data
        :input pass_id: string containing the satellite and ground station
        :input sod_annotation: integer representing the start of the anomalous sequence
        :input classification_bool: classification of each time step
        :input classification_confidence: confidence of the classification at each time step
        """
        # combined "detection evaluation" plot
        fig, axs = plt.subplots(3, sharex=False, sharey=False, figsize=(12, 4))
        fig.tight_layout()
        fig.suptitle('Day of Earthquake Predictions by Second of Day (SoD) for ' + pass_id + "\n")

        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])

        axs[0] = plt.subplot(gs[0])
        # TODO: NOTE the adjustment below for the window size, make dynamic
        sns.lineplot(data=event.iloc[59:, :], x="sod", y=pass_id, ax=axs[0])
        axs[0].set(yticklabels=[])
        axs[0].axvline(x=sat_annotation["start"], linestyle="dotted")  # start
        axs[0].axvline(x=sat_annotation["finish"], linestyle="dotted")  # approx end - 30 minutes later
        # TODo: do we need to revisit data generation / labeling?
        axs[0].margins(x=0)
        axs[0].set_xlabel("")

        axs[1] = plt.subplot(gs[1])
        sns.heatmap(np.array([classification_bool]), cbar=False, cmap="plasma", xticklabels=False, yticklabels=False,
                    ax=axs[1])
        axs[1].set_xlabel("Prediction (yellow is anomaly)")

        axs[2] = plt.subplot(gs[2])
        sns.heatmap(np.array([classification_confidence]), cbar=False, cmap="plasma_r", xticklabels=False,
                    yticklabels=False, ax=axs[2])  # .replace(0, np.nan)
        axs[2].set_xlabel("Prediction Confidence (yellow is worse)")

        plt.subplots_adjust(left=0.125,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.5)

        plt.savefig(save_path + "/" + pass_id + "_classification_plot.png")

    # TODO:
    def _out_of_sample(self, ground_truth_labels: dict, verbose: bool = False, save_path: Union[str, Path] = "./output") -> None:

        # create the save_path dir if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # establish a logger
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        # subdue the progress bar as to not clog the stdout / cell output
        with self.model.learner.no_bar():

            # disable logging
            self.model.learner.no_logging()

            # get the directory paths for the validation set specified
            validation_directories = self._get_directories(
                dir_path=self.validation_data_path
            )

            # process each of the directories in the validation set
            for d in tqdm(validation_directories, file=tqdm_out, total=len(validation_directories), mininterval=10, disable=operator.not_(verbose)):

                try:

                    # get the satellite name, ground station name and create an id
                    sat_name = d.split("__")[1]
                    ground_station_name = d.split("__")[0]
                    pass_id = ground_station_name + "__" + sat_name

                    # get the image files in the directory
                    image_files = self._get_image_files(
                        dir_path=self.validation_data_path,
                        subdir_path=d
                    )

                    classification = list()
                    classification_confidence = list()
                    windows = list()
                    window_start = 0
                    window_end = 60 # TODO: hard coded currently, make dynamic
                    for img in image_files:

                        try:

                            # load in the image and predict the classification
                            prediction = self.model.learner.predict(img)

                            # store the classification and the window range
                            classification.append(prediction[0])
                            classification_confidence.append(np.max(prediction[2].cpu().detach().numpy()))

                            windows.append([window_start, window_end])
                            window_start += 1
                            window_end += 1
                        except Exception as e:

                            print("Error encountered when predicting!")
                            if e is KeyboardInterrupt:
                                break

                    classification_bool = [0 if x == "normal" else 1 for x in classification]

                    # # store the classification result in a time-indexed array that is T minutes long by W windows tall
                    # # the number of minutes T is the number of windows plus the window size, or window_end
                    # period_classification_df = self._period_class(
                    #     len(image_files), self.windows, self.classification, self.classification_confidence
                    # )

                    # now we need to load in the original data (float data) that contains the second of day
                    # and other data needed for visualization and metrics reporting
                    df = self._read_data(ground_station_name, sat_name)

                    # get the day of year of the period for use with the ground truth
                    doy = datetime.datetime.utcfromtimestamp(df.index.values[0].tolist() / 1e9).timetuple().tm_yday # assumes period is entirely contained within a day

                    # identify continuous periods as we do when we generate the images and prep the data
                    events = np.split(df, np.where(np.isnan(df))[0])
                    events = [ev[~np.isnan(ev)] for ev in events if not isinstance(ev, np.ndarray)]
                    events = [ev.dropna() for ev in events if not ev.empty and ev.shape[0] > 100] # NOTE: 100 minute filter to remove short periods

                    # like the code that generates the "events", we will determine the predicted
                    # sequence of anomalies and record whether or not they are true positives

                    # For simplicity, we do not make scoring adjustments based on
                    # how early an anomaly was detected or the distance between false
                    # positives and labeled regions
                    ground_truth = [
                        ground_truth_labels[str(doy)][sat_name]["start"],
                        ground_truth_labels[str(doy)][sat_name]["finish"]
                    ]

                    # for now assume events is length 1 # TODO fix later
                    event = events[0].reset_index()

                    ground_truth_sequence = event[
                        (event["sod"] >= ground_truth[0]) & (event["sod"] <= ground_truth[1])].index.values

                    # adjust the sequence for the window size used earlier tp generate the images
                    # TODO: hard coded, make dynamic
                    adjusted_ground_truth_sequence = [x - 59 for x in ground_truth_sequence]

                    # get the indices of the anomalous values
                    anom_idx = np.where(np.array(classification_bool) == 1)
                    # get the sequences of the anomalous values
                    anom_sequences = Transforms.group_consecutives(list(anom_idx[0]))

                    # Obtain the true positives, false negatives, and false positives

                    tp, fn, fp = self._confusion_matrix_classification(
                        adjusted_ground_truth_sequence,
                        anom_sequences
                    )

                    self.tp += tp
                    self.fn += fn
                    self.fp += fp

                    # make pretty plots!
                    if verbose:
                        self._plot_classification(
                            events[0],
                            pass_id,
                            ground_truth_labels[sat_name],
                            classification_bool,
                            classification_confidence
                        )

                    if save_path is not None:
                        print('Saving File of: ', pass_id)
                        try:
                            save_df = event.iloc[59:].copy()
                            save_df['anomaly'] = classification_bool
                            save_df['confidence'] = classification_confidence
                            save_df.to_csv(save_path + "/" + pass_id + '_results.csv')
                        except FileNotFoundError:
                            print('Save Path "' + save_path + "/" + pass_id + '_results.csv' + '" not valid')

                except Exception as e:

                    if e is KeyboardInterrupt:
                        break

                    logging.warning(RuntimeWarning, str(e))

                    continue

            # calculate validation metrics
            precision = precision_score(self.tp, self.fp)
            recall = recall_score(self.tp, self.fn)
            f_score = f1_score(precision, recall)

            precision = self.exp.metric("validation_precision", precision)
            recall = self.exp.metric("validation_recall", recall)
            f_score = self.exp.metric("validation_f1_score", f_score)

            tp_sequence_length = np.mean(self.tp_lengths)
            tp_sequence_length = self.exp.metric("tp_sequence_length", tp_sequence_length)

            fp_sequence_length = np.mean(self.fp_lengths)
            fp_sequence_length = self.exp.metric("fp_sequence_length", fp_sequence_length)

            # if verbose, plot the distribution of the sequence lengths
            if verbose is True:
                self._plot_distribution(self.tp_lengths, self.fp_lengths)

    def run(self, ground_truth_labels: dict, verbose: bool = False, save_path: Union[str, Path] = "./output") -> None:

        """
        Runs the Experiment according to the specified Experiment and Model
        parameters.
        :param ground_truth_labels: A dictionary that contains ground truth start and
        end times (in second of day format) for specific days of the year and satellites.
        :param verbose: If True, shows Model training information and loss curves and plots
        results visually for interpretation.
        :param save_path: A specified file path in which to save Experiment artifacts.
        :return: None
        """

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

        # as part of the Experiment, perform an out-of-sample (OOS) validation of the results
        self._out_of_sample(
            ground_truth_labels=ground_truth_labels,
            save_path=save_path
        )

        # end the experiment
        self.exp.end()




