from matplotlib import backends
import tensorflow as tf
import numpy as np
from tomli import load
import mrcnn.model as modellib # https://github.com/matterport/Mask_RCNN/
from mrcnn.config import Config
import tensorflow.keras.backend as keras
from samples.leafs import leafs
import os

PATH_TO_SAVE_FROZEN_PB ="./"
FROZEN_NAME ="saved_model.pb"

def load_model(Weights):
        config = leafs.LeafsConfig()
        global model, graph
        class InferenceConfig(config.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        Weights = Weights
        Logs = "./logs"
        model = modellib.MaskRCNN(mode="inference", config=config,
		                  model_dir=Logs)
        model.load_weights(Weights, by_name=True)
        #graph = tf.compat.v1.get_default_graph()

        return model.keras_model

model = load_model("./mask_rcnn_leafscollage.h5")
ROOT_DIR = os.path.abspath("./")
saved_model_dir = os.path.join(ROOT_DIR, "mask_rcnn/1/")

#tf.saved_model.save(model,saved_model_dir)

# Convert the model
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)

tflite_model = converter.convert()

# Save the model.
with open(os.path.join(ROOT_DIR,"model.tflite"), 'wb') as f:
  f.write(tflite_model)
