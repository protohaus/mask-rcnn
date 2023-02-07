#%%
from fileinput import filename
import os
import sys
import random
import json
import math
from unittest import TestSuite
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import scipy

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

# def resize_and_crop(mask,dims):
#     height, width = mask.shape[:2]
#     target_height = dims[0]
#     target_width = dims[1]

#     if target_height > target_width:
#         scale = target_height / height
#     else:
#         scale = target_width / width

#     #scale mask
#     mask = scipy.ndimage.zoom(mask, zoom=[scale, scale], order=0)

#     new_height, new_width = mask.shape[:2]
#     y = int((new_height - target_height)/2)
#     x = int((new_width - target_width)/2)

#     #crop mask
#     crop_msk = mask[y:y+target_height, x:x+target_width]

#     return crop_msk

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
            NUM_CLASSES = 1 + 2

            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 800
            IMAGE_MAX_DIM = 1024

            # Skip detections with < 90% confidence
            DETECTION_MIN_CONFIDENCE = 0.8

            USE_MINI_MASK = False

        self.config = TestConfig()
        self.config.display()
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
                #resized_and_cropped = resize_and_crop(input_mask,dims)
                contours, hierarchy = cv2.findContours(input_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
                for count in contours:
                    epsilon = 0.002 * cv2.arcLength(count, True)
                    approximations = cv2.approxPolyDP(count, epsilon, True)
                    all_points_x = approximations[:,:,0]
                    all_points_y = approximations[:,:,1]
                    reshaped_x = all_points_x.flatten()
                    reshaped_y = all_points_y.flatten()
                    if len(reshaped_x) < 3:
                        continue
                    region = {"shape_attributes":{"name":"polygon","all_points_x":reshaped_x.tolist(),"all_points_y":reshaped_y.tolist()},           
                                "region_attributes": {
                                "Type": "Leaf",
                                "State": "healthy",
                                "Sort": "Genovese",
                                "Age": "medium",
                                "leaftinder": "undecided"
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

    def validate(self,TESTSET_INPUT_DIR,results):
        dataset_test = leafs_collage.LeafsCollageDataset()
        dataset_test.load_leafs(TESTSET_INPUT_DIR, "val")
        dataset_test.prepare()
        dataset = dataset_test

        # image_id is filename
        for i,name in enumerate(self.filenames):
            img, image_meta, gt_class_ids, gt_boxes, gt_masks = modellib.load_image_gt(dataset_test, self.config, dataset_test.image_from_source_map["leafs." + name])
            scaled_masks = np.zeros([1024, 1024, self.results[i][0]['masks'].shape[2]],
                dtype=np.bool)
            img, window, scale, padding, crop = utils.resize_image(self.images[i],max_dim = 1024)

            for j in range(self.results[i][0]['masks'].shape[2]):
                scaled_masks[:,:,j] = utils.resize_mask(self.results[i][0]['masks'][:,:,[j]],scale,padding)[:,:,0]
            
            mAP, precisions, recalls, overlaps = utils.compute_ap(gt_boxes,gt_class_ids,gt_masks,self.results[i][0]['rois'],self.results[i][0]['class_ids'],self.results[i][0]['scores'],scaled_masks)
            results["mAP"].append(mAP)
            results["precisions"].append(precisions)
            results["recalls"].append(recalls)
            results["overlaps"].append(overlaps)
        #return mAP, precisions, recalls, overlaps

    def activations(self,TEST_INPUT_FILE,layer):
        image = cv2.imread(TEST_INPUT_FILE)
        # Get activations of a few sample layers
        activations = self.model.run_graph([image], [
            ("input_image",        tf.identity(self.model.keras_model.get_layer("input_image").output)),
            (layer,          self.model.keras_model.get_layer("res4w_out").output),  # for resnet100
            ("rpn_bbox",           self.model.keras_model.get_layer("rpn_bbox").output),
            ("roi",                self.model.keras_model.get_layer("ROI").output),
        ])

        # Backbone feature map
        resulting = np.transpose(activations[layer][0,:,:,:4], [2, 0, 1])
        return resulting

def batch_inference(MODEL_DIR,LEAFS_MODEL_PATH,BATCH_INPUT_DIR):
    #TODO: create checks for valid input
    batch = AIBatchInference(MODEL_DIR,LEAFS_MODEL_PATH,BATCH_INPUT_DIR)
    batch.prepare_inference()
    batch.do_inference()
    batch.save_results()

def batch_validation(MODEL_DIR,LEAFS_MODEL_PATH,TESTSET_INPUT_DIR,results):
    batch = AIBatchInference(MODEL_DIR,LEAFS_MODEL_PATH,os.path.join(TESTSET_INPUT_DIR,"val"))
    batch.prepare_inference()
    batch.do_inference()
    results["mAP"] = []
    results["precisions"] = []
    results["recalls"] = []
    results["overlaps"] = []
    batch.validate(TESTSET_INPUT_DIR,results)

def show_activation(MODEL_DIR,LEAFS_MODEL_PATH,BATCH_INPUT_DIR,layer):
    batch = AIBatchInference(MODEL_DIR,LEAFS_MODEL_PATH,BATCH_INPUT_DIR)
    batch.prepare_inference()
    results = batch.activations(BATCH_INPUT_DIR,layer)
    return results

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
