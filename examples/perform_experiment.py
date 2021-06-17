# imports
from fastai.vision.all import Adam, resnet34
from tidd.modeling import Experiment, Model


# define a model
M = Model(
    architecture=resnet34,
    batch_size=256,
    learning_rate=0.0001,
    optimization_function=Adam
)

# specify source of data (in this case raw data to be converted to images)
training_data_paths = [
    "../data/hawaii_test"
]
validation_data_paths = [
    "../data/chile_test"
]

# define an experiment
E = Experiment(
    name="tidd-test",
    model=M,
    training_data_paths=training_data_paths,
    validation_data_paths=validation_data_paths, # optional, when ignored does not perform / allow Out of Sample
    share_testing=0.2,
    parallel_gpus=False,
    max_epochs=50,
    generate_data=True, # when true, uses paths params for raw. When false, uses paths params to mark imaged data.
    save_path="../output",
    window_size=60
)

# # run an experiment
# E.run(
#     verbose=True,
#     save_path="/home/vconstan/projects/tidd/output"
# )

