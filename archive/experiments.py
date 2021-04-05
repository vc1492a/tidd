from src.modeling import *

class Experiment(object):
    
    def __init__(self,
                 independent_variables: list,
                 satellites: list,
                 ground_stations: list,
                 model: 'tabular_learner',
                 year: int = 2012,
                 location: str = "hawaii",
                 elevation_filter: Union[int, None] = None,
                 time_aggregation: str = "1min",
                 batch_normalization: bool = True, weight_decay: float = 0.1,
                 output_save_directory: str = "", models_root_name: str = "model-latest",
                 max_epochs: int = 500, cuda_device: int = 0
                ):

        # experiment setup
        self.year = year
        self.location = location
        self.data_paths = Path('../data/' + self.location + '/' + str(self.year))
        self.days = [str(f).split("/")[-1] for f in self.data_paths.iterdir() if f.is_dir()]
        self.independent_variables = independent_variables
        self.satellites = satellites
        self.ground_stations = ground_stations
        self.elevation_filter = elevation_filter
        self.time_aggregation = time_aggregation
        self.batch_normalization = batch_normalization
        self.weight_decay = weight_decay

        #Create an output folder with naming structure(SatExperiment_with_Data_[location]_[Year])
        save_dir = output_save_directory + '/SatExperiment_' + self.location + '_' + str(self.year) + '/output'
        save_dir_path = Path(save_dir)
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        cv_log_cols = [
                "ground station + satellite",
                "root_mean_square_error", 
                "period", 
                "learn_rate", 
                "training_time_seconds",
                "gpu_memory_bytes"
            ]
                
        information_df = pd.DataFrame(columns = cv_log_cols).set_index("ground station + satellite")
        information_df.to_csv(Path(str(save_dir_path) + "/" + "training_log.csv"))

        self.save_dir = save_dir
        self.models_root_name = models_root_name
        self.max_epochs = max_epochs
        
        # set the cuda device for experiment
        torch.cuda.set_device(cuda_device)
        self.cuda_device = torch.device("cuda:" + str(cuda_device))
       
        
       
        self.model = model
        self.dataframes = dict()  # stores individual dataframes read from csv
        
        # This was just for me to know if the cuda device is set up correctly
        print("For this Experiment:")
        print("Using following CUDA device:", torch.cuda.get_device_name(self.cuda_device))
        print("Output will be located here: ", self.save_dir)
        
        # experiment tracking - one entry per ground station and satellite combination 
        self.results = dict()
    
    
    #TODO: migrate this to Data class
    def prep_data(self) -> None: 
        """
        Data is split into periods where any NaN values are removed. 
        These are then normalized and made into a dataset that can be used 
        for modeling (training, test, and validation sets.)
        """
        self.dataframes = Data.read_data(
            ground_stations=self.ground_stations,
            satellites=self.satellites,
            location=self.location,
            year=self.year,
            days=self.days,
            elevation_filter=self.elevation_filter,
            time_aggregation=self.time_aggregation
            )
        
        print("Finished reading data . . .")

        # first, remove NaNs and split the data into periods 
        for station_sat in tqdm(self.dataframes["all_days"].keys()):
            
            try:
                        
                events = Data.get_periods(self.dataframes["all_days"][station_sat]["merged_dataframe"])        

                # store the events for that 
                self.dataframes["all_days"][station_sat]["periods"] = events

                # then, rescale the values from -1 to 1 for each period to support the modeling 
                self.dataframes["all_days"][station_sat]["periods_normalized"] = Data.rescale_periods(events=events, feature_range=(-1,1))

                # allocate a training set, a testing set, and a validation set 
                self.dataframes["all_days"][station_sat]["periods_allocated"] = Data.allocate_periods(periods=self.dataframes["all_days"][station_sat]["periods_normalized"], index_earthquake=12)

                # TODO: above allocation function may not work well for all since period based 
                # may be better to do so off of a specified time range within a period 
            
            except Exception as ex:
                print("ERROR: Unable to prep data for " + station_sat)
                

    def train_models(self, input_layers: list, input_ps: list, batch_size: int, verbose: bool = False) -> None:
        """
        For each ground station and satellite combination, train a model and 
        track the associated artifacts, output, and session information (such as the 
        training time). 
        """
        training_log = str(Path(self.save_dir)) + '/training_log.csv'
        for station_sat in tqdm(self.dataframes["all_days"].keys()):
            
            # reset the starting point in tracking max GPU memory occupied
            torch.cuda.reset_max_memory_allocated()

            
            #create folder from which to save models and modeling
            model_save_dir = self.save_dir + '/' + station_sat

            print(model_save_dir)

            if not os.path.exists(model_save_dir):
                Path(model_save_dir).mkdir(parents=True, exist_ok=True)

            # create an entry in the results for later tracking 
            self.results[station_sat] = dict()

            # get the data to be modeled from the dictionary 
            data_to_model = self.dataframes["all_days"][station_sat]["periods_allocated"]

            # create the training, test, and valdiation sets to use in model training
            # we first need to assign the dependent variable based on the 
            df_train = pd.concat(data_to_model["train"])
            df_train[station_sat + "_target"] = df_train[station_sat]
            df_test = data_to_model["test"]
            df_test[station_sat + "_target"] = df_test[station_sat]
            df_valid = data_to_model["valid"]
            df_valid[station_sat + "_target"] = data_to_model["valid"][station_sat]

            #create the data bunch to be used in the modeling process
            data_bunch = Model.make_data_bunch(
                df_train,
                df_test,
                df_valid,
                [station_sat + var for var in self.independent_variables],
                station_sat + "_target", 
                cuda_device=self.cuda_device,
                include_catvars=False, 
                catvars=None, 
                batch_size=batch_size
            ) 

            # store the data bunch of the model for later use by predict functions 
            self.dataframes["all_days"][station_sat]["data_bunch"] = data_bunch


            #create the learner
            lr = tabular_learner(
                data_bunch["databunch"], 
                layers=input_layers, 
                ps=input_ps,
                path=model_save_dir,
                metrics=[root_mean_squared_error], 
                callback_fns=[CSVLogger, PeakMemMetric],
                use_bn=self.batch_normalization,
                wd=self.weight_decay
            )

            # needed some way to include the cuda device
            # first, identify the ideal learning rate for this set of data 


            learn_rate = Model.find_appropriate_lr(model=lr,loss_threshold=0.001, plot=verbose) 
            print("Learning Rate for {}:".format(station_sat), learn_rate)
            if learn_rate > 0.01:
                warnings.warn("Learning Rate may be set too high. Current Learning rate is {}".format(learn_rate))
            # since the learning rate can vary from model to model, let's track it 
            self.results[station_sat]["learning_rate"] = learn_rate

            # start a timer to track the training time 
            start = time.time()

            # train the model according to the specified architecture, paremeters, and 
            # identified learning rate 
            # TODO: right now that information is stored in the experiment class
            # but would be nicer if specified in the model class 
            # train the model 
            lr.fit_one_cycle(
                self.max_epochs,
                learn_rate,
                # the save model callback saves the model after each epoch 
                # the early stopping callback helps avoid overfitting and monitors the validation loss 
                callbacks=[
                    SaveModelCallback(
                        lr, 
                        every='epoch', 
                        monitor=['accuracy', 'root_mean_square_error']
                    ),
                    EarlyStoppingCallback(
                        lr,
                        monitor='valid_loss', #'valid_loss', 'root_mean_square_error'
                        min_delta=0.0001, 
                        patience=15
                    ),
                    ShowGraph(
                        lr
                    )
                ]
            )

            # end a timer to track the training time and add to the results dict 
            finish = time.time()
            training_time = finish - start
            self.results[station_sat]["training_time"] = training_time




            # TODO (diff from above): store the losses and metric plots for each model 
#             # loaded learners do not have a recorder
            if verbose:
                lr.recorder.plot_losses()
                lr.recorder.plot_metrics()

            # rmse, learner, time_diff
            #time_series_cv_log = list()
            #cv_log_cols = ["root_mean_square_error", "period", "learn_rate", "training_time_seconds"]
            # TODO: find a way to store the gpu memory and gpu utilization



            # store the root mean square error for this model 
            rmse = lr.recorder.metrics[-1][0].item()
            self.results[station_sat]["root_mean_square_error"] = rmse
            bytes_allocated = torch.cuda.max_memory_allocated()
            training_cv_log = [
                station_sat, 
                rmse, 
                0, 
                learn_rate, 
                training_time, 
                bytes_allocated
            ]
            with open(training_log, "a+") as fd:
                writer = csv.writer(fd)
                writer.writerow(training_cv_log)

            # TODO: store the training/testing and validation loss, num epochs also 

            # save the model for this specific station and satellite combination 
            # TODO: maybe store the model in the output dir, e.g. output/models
            save_location = model_save_dir + "/" + self.models_root_name + "-" + station_sat

            lr.save("")

            # TODO fix export
            lr.export('-export.pkl')

            lr.show_results()



            # store the trained learner in the experiment class for use in prediction and classification calls by the experiment class 
            self.results[station_sat]["trained_learner"] = lr 
                
#             except Exception as ex:
#                 print("ERROR: Unable to train model for " + station_sat)
                
            
    def predict_models(self) -> None: 
        """
        For each ground station and satellite combination, predicts the values for 
        sTEC d/dT on the day of the earthquake. Predictions and residual values are 
        stored for later visualization and analysis. 
        """
        
        # for each model 
        for station_sat in tqdm(self.dataframes["all_days"].keys()):
            
            try:
            
                # taking the learner and the associated data bunch and trained learner
                data_bunch = self.dataframes["all_days"][station_sat]["data_bunch"]
                lr = self.results[station_sat]["trained_learner"]

                # predict on the test data and store predictions
                df_test = Model.predict_values(
                    dataframe=data_bunch["test"], 
                    learner=lr, 
                    dependent=station_sat + "_target",
                    frac=1.
                ).sort_index()

                # predict on the validation data (day of earthquake) and store predictions 
                df_valid = Model.predict_values(
                    dataframe=data_bunch["valid"], 
                    learner=lr, 
                    dependent=station_sat + "_target",
                    frac=1.
                ).sort_index()

                # store the residuals in a directory for later analysis, further review for both test and validation sets 
                # if a directory doesn't exist for the model, create it 
                # TODO: below hard-coded path will need to be tweaked when pulling into tidd
                #       maybe allow for dynamic input so that the output does not stay within the repository (Hamlin )
                output_path = self.save_dir + '/' + station_sat + "/residuals"
                if not os.path.exists(output_path):
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                # for the test set and valid set, store the prediction dataframes
                # these dataframes contain the residuals (errors) raw and absolute 
                df_test.to_csv(output_path + "/test.csv")
                df_valid.to_csv(output_path + "/valid.csv")
                
            except Exception as ex:
                print("ERROR: Unable to make predictions for " + station_sat)
                
        
        
    # TODO: an experiment "run" function could call train, predict, metrics calling etc .
    # that way, much of train code and predict code can be put into the model class 
    # metrics class sepearate, too. Called for each model probably. 
    # would make it really easy for the user: create experiment class, pass models (which have associated data), get results 
    # I think we would want to remove tqdm from elsewhere in that case and show 
    # a progress bar for each model that covers train, predict, classify (former is a to-do once we handle residual values)
            
            
            
            
        
        
        
        
        
        
        
            
            
            