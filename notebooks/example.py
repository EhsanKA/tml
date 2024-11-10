from tml.model.pipeline import Pipeline, ModelHandler
from tml.models.model import BinaryClassificationLightning
from tml.data_prep.load_data import load_and_preprocess_data

# Prepare the data
cols = range(1, 42+1)
input_dim = len(cols) - 1
input_path = '../data/test1.csv'
load_dict =load_and_preprocess_data(input_path, cols)
dataset = load_dict['test_set']
data = dataset[:, 1:]
hard_targets = dataset[:, 0]


model_instance = BinaryClassificationLightning(input_dim=41, nb_classes=1, dropout_rate=0.5, learning_rate=0.001)

# OR

model_class = BinaryClassificationLightning
model_config = {
    "input_dim": 41,
    "nb_classes": 1,
    "dropout_rate": 0.5,
    "learning_rate": 0.001
}

# Initialize the pipeline
model_handler = ModelHandler(model_instance=model_instance, model_class=model_class, model_config=model_config)
pipeline = Pipeline(model_handler, data, hard_targets, max_epochs=2)



pipeline.run(n_steps=2)
# pipeline.evaluate()
