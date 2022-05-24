#%%
import os
import sys
import random
import json
import math
import cv2
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

print(ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib

from samples.leafs_collage import leafs_collage

class AIBatchInference():
    def __init__(self,MODEL_DIR,LEAFS_MODEL_PATH,BATCH_INPUT_DIR):
        self.MODEL_DIR = MODEL_DIR
        self.LEAFS_MODEL_PATH = LEAFS_MODEL_PATH
        self.BATCH_INPUT_DIR = BATCH_INPUT_DIR
        self.config = leafs_collage.LeafsCollageConfig()

        class TestConfig(self.config.__class__):
            NAME = "test"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + 1

            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 800
            IMAGE_MAX_DIM = 1024

            # Skip detections with < 90% confidence
            DETECTION_MIN_CONFIDENCE = 0.8

        config = TestConfig()
        config.display()
        self.Device = "/cpu:0"
        self.TEST_MODE = "inference"
    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

    def prepare_inference(self):
        # Create model in inference mode
        with tf.device(self.Device):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR,
                                    config=self.config)
        weights_path = self.LEAFS_MODEL_PATH

        # Or, load the last model you trained
        # weights_path = model.find_last()

        # Load weights
        print("Loading weights ", weights_path)
        self.model.load_weights(weights_path, by_name=True)

    def do_inference(self):
        self.images = []
        self.filenames = []
        for image in os.listdir(self.BATCH_INPUT_DIR):
            if image.lower().endswith('.jpg'):
                img = cv2.imread(os.path.join(self.BATCH_INPUT_DIR,image))
                self.images.append(img)
                self.filenames.append(image)

        self.results = []
        # Run object detection
        for image in self.images:
            result = self.model.detect([image], verbose=1)
            self.results.append(result)

    def save_results(self):
        annotations = {}
        for i, res in enumerate(self.results):
            now = datetime.now()

            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d%m%Y_%H%M%S")
            #plt.imshow(bg_image)
            filename = self.filenames[i]
            regions = []

            for j in range(res[0]['masks'].shape[2]):
                mask = res[0]['masks'][:,:,j]
                input_mask = mask.astype(np.uint8)
                contours, hierarchy = cv2.findContours(input_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
                for count in contours:
                    epsilon = 0.002 * cv2.arcLength(count, True)
                    approximations = cv2.approxPolyDP(count, epsilon, True)
                    all_points_x = approximations[:,:,0]
                    all_points_y = approximations[:,:,1]
                    reshaped_x = all_points_x.flatten()
                    reshaped_y = all_points_y.flatten()
                    region = {"shape_attributes":{"name":"polygon","all_points_x":reshaped_x.tolist(),"all_points_y":reshaped_y.tolist()},           
                                "region_attributes": {
                                "Type": "Leaf",
                                "State": "healthy",
                                "Sort": "Genovese",
                                "Age": "medium"
                            }}
                    regions.append(region)

            filesize = os.path.getsize(os.path.join(self.BATCH_INPUT_DIR,filename))

            annotations[filename + str(filesize)] = {"filename":filename,"size":filesize, "regions":regions,"fileattributes":{}}

        now = datetime.now()

        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d%m%Y_%H%M%S")
            #plt.imshow(bg_image)
        with open(os.path.join(self.BATCH_INPUT_DIR,'via_region_data{0}.json'.format(dt_string)), 'w') as f:
            json.dump(annotations, f)

def batch_inference(MODEL_DIR,LEAFS_MODEL_PATH,BATCH_INPUT_DIR):
    #TODO: create checks for valid input
    batch = AIBatchInference(MODEL_DIR,LEAFS_MODEL_PATH,BATCH_INPUT_DIR)
    batch.prepare_inference()
    batch.do_inference()
    batch.save_results()


# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#LEAFS_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_leafscollage.h5")

#config = leafs_collage.LeafsCollageConfig()
#BATCH_INPUT_DIR = os.path.join(ROOT_DIR, "datasets/leafs_collage")
#DATA_DIR = os.path.join(ROOT_DIR,"inference","data")

#config = leafs_collage.LeafsCollageConfig()
#LEAF_DIR = os.path.join(ROOT_DIR, "datasets/leafs_collage")

# Override the training configurations with a few
# changes for inferencing.
# define the test configuration


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
#DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
#TEST_MODE = "inference"

# Load validation dataset
# dataset = leafs_collage.LeafsCollageDataset()
# dataset.load_leafs(LEAF_DIR, "val")

# Must call before using the dataset
# dataset.prepare()

# print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))



# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
#weights_path = LEAFS_MODEL_PATH

# Or, load the last model you trained
# weights_path = model.find_last()

# Load weights
#print("Loading weights ", weights_path)
#model.load_weights(weights_path, by_name=True)

""" images = []
filenames = []
for image in os.listdir(DATA_DIR):
    if image.lower().endswith('.jpg'):
        img = cv2.imread(os.path.join(DATA_DIR,image))
        images.append(img)
        filenames.append(image)

results = []
# Run object detection
for image in images:
    result = model.detect([image], verbose=1)
    results.append(result) """

# Display results
""" ax = get_ax(1)
r = results[0]
visualize.display_instances(images[0], r[0]['rois'], r[0]['masks'], r[0]['class_ids'], 
                            ["background","leafs"], r[0]['scores'], ax=ax,
                            title="Predictions") """

#store results as json
# %% create mask for each leaf.
""" annotations = {}
for i, res in enumerate(results):
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    #plt.imshow(bg_image)
    filename = filenames[i]
    regions = []

    for j in range(res[0]['masks'].shape[2]):
        mask = res[0]['masks'][:,:,j]
        input_mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(input_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
        for count in contours:
            epsilon = 0.002 * cv2.arcLength(count, True)
            approximations = cv2.approxPolyDP(count, epsilon, True)
            all_points_x = approximations[:,:,0]
            all_points_y = approximations[:,:,1]
            reshaped_x = all_points_x.flatten()
            reshaped_y = all_points_y.flatten()
            region = {"shape_attributes":{"name":"polygon","all_points_x":reshaped_x.tolist(),"all_points_y":reshaped_y.tolist()},           
                        "region_attributes": {
                        "Type": "Leaf",
                        "State": "healthy",
                        "Sort": "Genovese",
                        "Age": "medium"
                    }}
            regions.append(region)

    filesize = os.path.getsize(os.path.join(DATA_DIR,filename))

    annotations[filename + str(filesize)] = {"filename":filename,"size":filesize, "regions":regions,"fileattributes":{}}

now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d%m%Y_%H%M%S")
    #plt.imshow(bg_image)
with open(os.path.join(DATA_DIR,'via_region_data{0}.json'.format(dt_string)), 'w') as f:
    json.dump(annotations, f) """
