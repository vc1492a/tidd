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
import glob
from hyperdash import Experiment as HyperdashExperiment
import logging
import natsort
import numpy as np
import operator
import os
from pathlib import Path
import sys
from tidd.metrics import confusion_matrix_scores, calculating_coverage, \
    precision_score, recall_score, f1_score, confusion_matrix_classification
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
        :output: class label
        :output: np.ndarray of confidence values
        """
        prediction = self.learner.predict(test_item)
        return prediction[0], prediction[2].cpu().detach().numpy()

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

        # # TODO: initialize an object in which to store metrics
        # self.metrics = dict()

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
        # TODO
        """

        # define the save path for the out of sample output and make sure the path exists
        
        save_path = self.save_path + '/' + 'out_of_sample'
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # probably need 

        # subdue the progress bar as to not clog the stdout / cell output
        with self.model.learner.no_bar():
            # disable logging
            self.model.learner.no_logging()

            # initialize some metrics counting
            # TODO: not sure if you wanted this out of the function completely
            self.tp = 0
            self.fn = 0 
            self.fp = 0 
            self.tp_lengths = list()
            self.fp_lengths = list()

            # get the full path of each directory containing image files
            image_directories = Data._get_image_directories(self.validation_data_path)
            # filter for those containing "unlabeled"
            image_directories = [i for i in image_directories if "unlabeled" in i]
            logging.info(image_directories)

            # establish a logger
            tqdm_out = TqdmToLogger(logger, level=logging.INFO)
            
            # TODO (future work): parallel process the below
            # process each of the directories in the validation set
            for d in tqdm(image_directories, file=tqdm_out, total=len(image_directories), mininterval=10, disable=operator.not_(verbose)):
            
                logging.info(d)
        
                # get the images in the directory 
                image_files = [d + "/" + f for f in
                        natsort.natsorted(os.listdir(d)) if ".jpg" in f and
                        f[0] != "."]
            
                logging.info(len(image_files))
                #logging.info(image_files)

                # for the sorted images in the directory, predict the sequence
                try:
                    classification, classification_confidence, classification_bool = self.model.predict_sequences(image_files)
                except Exception as ex:
                    logging.warning(RuntimeWarning, "Error encountered when predicting sequence.")
                    logging.warning(str(ex))
                    if ex is KeyboardInterrupt:
                        break
                    continue
                    
                logging.info(len(classification))
                    
                # we need to load in the original data file (float data) that contains the second of day 
                # and other data needed for visualization and metrics 
                logging.info(image_files[0])
                #doy = str(image_files[0].split("/")[-1].split("_")[0])
                #sat_name = 



                # 
                
                




            #    # error capturing
            #    try:
            #
            #         # TODO: for each labeled day of year and satellite, get matching from validation_directories
            #         # will assume that the variable is called image_files
            #         # get the satellite name, ground station name and create an id
            #         sat_name = d.split("__")[1]
            #         ground_station_name = d.split("__")[0]
            #         pass_id = ground_station_name + "__" + sat_name
            #
            #         # get the image files in the directory
            #         base_path = dir_path + "/" + subdir_path + "/unlabeled"
            #         image_files = [base_path + "/" + f for f in natsort.natsorted(os.listdir(base_path)) if
            #                        ".jpg" in f and "302" in f.split("_")[0]] subdir_path=d
            #         )
            #
            #         # NOTE: start predict_sequence
            #         # TODO: currently assumes that image_files will contain the list of image_files
            #         # TODO: need to find a way to create list of the image paths as image_files variable
            #         # window creation does not depend on the model prediction
            #         # but only on the amount of windows/images to predict over
            #         #image_files = [] # TODO: placeholder
            #         #window_start = list(range(len(image_files)))
            #         #window_end = list(map(lambda x: x + self.window_size - 1, window_start))
            #         #windows = list(zip(window_start, window_end))
            #
            #         #try:
            #          #   classification, classification_confidence, classification_bool = self.model.predict_sequences(image_files)
            #         #except Exception as e:
            #         #    print("Error encountered when predicting!")
            #         #    if e is KeyboardInterrupt:
            #         #        break
            #
            #         # NOTE: end predict_sequence function
            #
            #         # NOTE: start generate ground truth sequences function
            #         # TODO: utilize functions in Data class to read data from file
            #         # # now we need to load in the original data (float data) that contains the second of day
            #         # # and other data needed for visualization and metrics reporting
            #
            #         # TODO: get the name/location of the file dynamically
            #         #ground_station_name = "ahup" # TODO: Placeholder
            #         #sat_name = "G07" # TODO: Placeholder
            #         #sat = "../data/hawaii/2012/302/" + ground_station_name + "3020.12o_" + sat_name + ".txt" # TODO: Placeholder
            #
            #
            #         #df = Data.read_data_from_file(sat)
            #         #df = Transform.sod_to_timestamp(df) # this should be the same as before now
            #
            #         #
            #         # # get the day of year of the period for use with the ground truth
            #         # doy = datetime.datetime.utcfromtimestamp(df.index.values[
            #         #                                              0].tolist() / 1e9).timetuple().tm_yday  # assumes period is entirely contained within a day
            #         #
            #         # # identify continuous periods as we do when we generate the images and prep the data
            #         # events = np.split(df, np.where(np.isnan(df))[0])
            #         # events = [ev[~np.isnan(ev)] for ev in events if not isinstance(ev, np.ndarray)]
            #         # events = [ev.dropna() for ev in events if
            #         #           not ev.empty and ev.shape[0] > 100]  # NOTE: 100 minute filter to remove short periods
            #         #
            #         # # like the code that generates the "events", we will determine the predicted
            #         # # sequence of anomalies and record whether or not they are true positives
            #         #
            #         # # For simplicity, we do not make scoring adjustments based on
            #         # # how early an anomaly was detected or the distance between false
            #         # # positives and labeled regions
            #         # ground_truth = [
            #         #     ground_truth_labels[str(doy)][sat_name]["start"],
            #         #     ground_truth_labels[str(doy)][sat_name]["finish"]
            #         # ]
            #         #
            #         # # for now assume events is length 1 # TODO fix later
            #         # event = events[0].reset_index()
            #         #
            #         # ground_truth_sequence = event[
            #         #     (event["sod"] >= ground_truth[0]) & (event["sod"] <= ground_truth[1])].index.values
            #         #
            #         # # adjust the sequence for the window size used earlier tp generate the images
            #         # # TODO: hard coded, make dynamic
            #         # adjusted_ground_truth_sequence = [x - self.window_size for x in ground_truth_sequence]
            #
            #         # NOTE end get ground truth sequence function
            #
            #         # TODO: get the sequences of the anomalous values (make this a function in Model class)
            #
            #         # TODO: get the true positives, etc.
            #         # TODO: assumes that adjusted_ground_sequence and anom_sequences have the same behavior in model.py
            #         #adjusted_ground_truth_sequence = [] # TODO: placeholder
            #         #anom_sequences = [] # TODO: placeholder
            #         #tp, fn, fp, sub_tp_lens, sub_fp_lens = confusion_matrix_classification(adjusted_ground_truth_sequence, anom_sequences)
            #
            #         #self.tp += tp
            #         #self.fn += fn
            #         #self.fp += fp
            #         #self.tp_lengths += sub_tp_lens
            #         #self.fp_lengths += sub_fp_lens
            #
            #         # TODO: fill in the parameters for plot_classification
            #         # TODO: might want to change the path
            #         #if verbose:
            #             #fig = plot_classification()
            #             #fig.savefig(save_path + '/')
            #
            #     except Exception as e:
            #
            #         # if keyboard interrupt break
            #         if e is KeyboardInterrupt:
            #             break
            #         # else continue
            #         logging.warning(RuntimeWarning, str(e))
            #         continue
            #
            # # TODO: calculate and report the validation metrics
            #
            #
            # # TODO: if verbose plot dist plot
            # # # check parameters
            # # if verbose is True:
            # #     ax = plot_distribution(self.tp_lengths, self.fp_lengths)
            # #     ax.savefig(save_path + "/classification_sequence_length_distribution.jpg")

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

        cm = interp.confusion_matrix()

        results = confusion_matrix_scores(cm)

        # track results in the Hyperdash experiment
        self.exp.metric("accuracy", results[0])
        self.exp.metric("precision", results[1])
        self.exp.metric("recall", results[2])
        self.exp.metric("F1 Score", results[3])

        # calculate the coverage
        predictions, targets = self.model.learner.get_preds()  # by default uses validation set
        anom_cov, normal_cov = calculating_coverage(predictions, targets)
        self.exp.metric("anomaly coverage", anom_cov)
        self.exp.metric("normal coverage", normal_cov)

        # if verbose, show results
        if verbose is True:

            # plot the results
            interp.plot_top_losses(9, figsize=(15, 11))
            interp.plot_confusion_matrix(figsize=(4, 4), dpi=120)

        # TODO: # # as part of the Experiment, perform an out-of-sample (OOS) validation of the results
        self._out_of_sample(verbose=verbose)

        # end the experiment
        self.exp.end()





