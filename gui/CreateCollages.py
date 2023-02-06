#%% Import packages
import cv2
import os
import random
from cv2 import rotate
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

#%% Pick random color
def pick_random_unique_color(colors):
    r = 0
    g = 0
    b = 0
    new_color = np.array([r,g,b])
    while np.equal(colors,new_color).any():
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        new_color = np.array([r,g,b])
    
    colors = np.append([colors],[new_color],axis=0)

    #print(colors)
    return colors

# There are three channels, we need 2 kinds of information: Which instance and which
# class: take blue for classes then there are 255 different classes possible,
# with 256*256 = 65536 instances per class, black is for background
def pick_random_unique_redgreen(colors,class_index):
    r = 0
    g = 0
    b = class_index
    new_color = np.array([r,g,b])

    subcolors = [color for color in colors[:] if (color[:][2] == class_index or color[:][2] == 0)]

    while np.equal(subcolors,new_color).any():
        r = random.randint(0,255)
        g = random.randint(0,255)
        new_color = np.array([r,g,b])
    
    colors = np.vstack([colors,new_color])
    #print(colors)
    return colors

def pick_random_grayscale(grays):
    g = 0

    new_gray = np.array([g])
    while np.equal(grays,new_gray).any():
        g = random.randint(0,255)
        new_gray = np.array([g])

    grays = np.append([grays],[new_gray],axis=0)

    return grays
#%% Image Manipulation Functions
def rotate_image(img, angle):
    #print("Original image shape:")
    #print(img.shape)

    height, width = img.shape[:2]
    image_center = (int(width/2), int(height/2))

    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0,0]) 
    abs_sin = abs(rot_mat[0,1])

    bbox_w = int(height * abs_sin + width * abs_cos)
    bbox_h = int(height * abs_cos + width * abs_sin)

    rot_mat[0, 2] += bbox_w/2 - image_center[0]
    rot_mat[1, 2] += bbox_h/2 - image_center[1]

    result = cv2.warpAffine(img.astype(np.uint8), rot_mat, (bbox_w, bbox_h), flags=cv2.INTER_NEAREST)
    
    #print("Rotated image shape:")
    #print(result.shape)
    return result

def flip_horizontal(img):
    result = np.flip(img, 1)
    return result

def shear(img):
    height, width, dim = img.shape
    # transformation matrix for Shearing
    # shearing applied to x-axis
    shear_x = random.uniform(0,0.1)
    shear_y = random.uniform(0,0.1)
    M = np.float32([[1, shear_y, 0],
                    [shear_x, 1  , 0],
                    [0, 0  , 1]])
    # shearing applied to y-axis
    # M = np.float32([[1,   0, 0],
    #             	  [0.5, 1, 0],
    #             	  [0,   0, 1]])
    # apply a perspective transformation to the image                
    sheared_img = cv2.warpPerspective(img,M,(int(width/(1-shear_y)),int(height/(1-shear_x))),flags=cv2.INTER_NEAREST)
    return sheared_img

def scale_random(img):
    # get the image shape
    height, width, dim = img.shape
    #transformation matrix for Scaling
    scale_x = random.uniform(0.9,1.5)
    scale_y = random.uniform(0.9,1.5)
    M = np.float32([[scale_x, 0  , 0],
            	[0,   scale_y, 0],
            	[0,   0,   1]])
    # apply a perspective transformation to the image
    scaled_img = cv2.warpPerspective(img,M,(int(scale_x*width),int(scale_y*height)),flags=cv2.INTER_NEAREST)
    return scaled_img

#%% Add Leaf to image
# add random leaf with random rotation, shear, scale and location
# To-Do: Scalierung ermöglichen

def overlay_image(img,img_overlay,img_mask,y_c,x_c,mask_colors,category):

    if y_c < 0 or x_c < 0 or y_c > img.shape[0] or x_c > img.shape[1]:
        return

    y = int(y_c - img_overlay.shape[0]/2)
    x = int(x_c - img_overlay.shape[1]/2)

    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    alpha_leafs = img_overlay[y1o:y2o, x1o:x2o, 3] / 255.0 # 0 ist transp
    alpha_bg = 1.0 - alpha_leafs # wo Bild transp ist, ist bg opak

    alpha_mask = alpha_leafs
    alpha_mask[alpha_mask > 0] = 1.0
    alpha_mask_bg = 1.0 - alpha_mask

    #mask_colors = pick_random_unique_color(mask_colors)
    mask_colors = pick_random_unique_redgreen(mask_colors,category)
    #gray_colors = pick_random_grayscale(mask_colors)

    mask_leaf = np.zeros([img_overlay.shape[0],img_overlay.shape[1],3],dtype=np.uint8)
    mask_leaf[:,:] = mask_colors[-1]

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_leafs * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_bg * img[y1:y2, x1:x2, c])

        img_mask[y1:y2, x1:x2,c] = (alpha_mask * mask_leaf[y1o:y2o, x1o:x2o, c] +
                              alpha_mask_bg * img_mask[y1:y2, x1:x2,c])
#%% create random image transformations
def random_transformation(image):
    rotated = rotate_image(image,random.uniform(-180.0,180.0))
    if random.randint(0,1) == 1:
        flipped = flip_horizontal(rotated)
    else:
        flipped = rotated
    sheared = shear(flipped)
    result = scale_random(sheared)

    return result

# %% create mask for each leaf.

# %%

def create_collages(LEAF_FOLDER,BG_FOLDER,OUTPUT_FOLDER, N):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(LEAF_FOLDER):
        print("Leaf folder not existing...")
        return
    if not os.path.exists(BG_FOLDER):
        print("Background folder not existing...")
        return
    healthy_path = os.path.join(LEAF_FOLDER,'healthy')
    withered_path = os.path.join(LEAF_FOLDER,'withered')
    if not os.path.exists(healthy_path):
        print("Healthy Leaf folder not existing...")
        return
    if not os.path.exists(withered_path):
        print("Withered Leaf folder not existing...")
        return

    leaf_images = {'healthy':[],'withered':[]}
    bg_images = []
    mask_images = []
    scale_percent = 35 # percent of original size
    scale_percent_leafs = 66

    for image in os.listdir(healthy_path):
        if image.lower().endswith('.png'):
            leaf_image = cv2.imread(os.path.join(healthy_path,image),flags=cv2.IMREAD_UNCHANGED)
            new_width = int(leaf_image.shape[1] * scale_percent_leafs / 100)
            new_height = int(leaf_image.shape[0] * scale_percent_leafs / 100)
            dim = (new_width, new_height)

            # resize image
            resized = cv2.resize(leaf_image, dim, interpolation = cv2.INTER_NEAREST)
            leaf_images['healthy'].append(resized)# and class
        else:
            print("Found non-png type in folder.")
    if len(leaf_images['healthy']) == 0:
        print("Loaded 0 leaf images")
        return

    for image in os.listdir(withered_path):
        if image.lower().endswith('.png'):
            leaf_image = cv2.imread(os.path.join(withered_path,image),flags=cv2.IMREAD_UNCHANGED)
            new_width = int(leaf_image.shape[1] * scale_percent_leafs / 100)
            new_height = int(leaf_image.shape[0] * scale_percent_leafs / 100)
            dim = (new_width, new_height)

            # resize image
            resized = cv2.resize(leaf_image, dim, interpolation = cv2.INTER_NEAREST)
            leaf_images['withered'].append(resized)# and class
        else:
            print("Found non-png type in folder.")
    if len(leaf_images['healthy']) == 0:
        print("Loaded 0 leaf images")
        return

    for image in os.listdir(BG_FOLDER):
        if image.lower().endswith('.jpg'):
            bg_image = cv2.imread(os.path.join(BG_FOLDER,image),flags=cv2.IMREAD_UNCHANGED)
            new_width = int(bg_image.shape[1] * scale_percent / 100)
            new_height = int(bg_image.shape[0] * scale_percent / 100)
            dim = (new_width, new_height)

            # resize image
            resized = cv2.resize(bg_image, dim, interpolation = cv2.INTER_NEAREST)
            bg_images.append(resized)
        else:
            print("Found non-jpg type in folder.")
    if len(bg_images) == 0:
        print("Loaded 0 background images")
        return

    annotations = {}
    cluster = True
    for i in range(N):
        bg_index = random.randint(0,len(bg_images)-1)
        bg_image = np.copy(bg_images[bg_index])
        mask_image = np.zeros([bg_image.shape[0],bg_image.shape[1],3],dtype=np.uint8)
        regions = []
        colors = np.array([[0,0,0]])
        #colors = np.array([0])
        cluster_scale = int(bg_image.shape[0]/3)
        if cluster:
            x_cluster = random.randint(cluster_scale,bg_image.shape[0]-cluster_scale)
            y_cluster = random.randint(cluster_scale,bg_image.shape[1]-cluster_scale)

        for j in range(random.randint(25,100)):
            category = random.randint(1,2)
            key = 'healthy'
            if category == 2:
                key = 'withered'
            if cluster:
                x_c = x_cluster + random.randint(-cluster_scale,cluster_scale)
                y_c = y_cluster + random.randint(-cluster_scale,cluster_scale)
                if x_c > bg_image.shape[0]:
                    x_c = bg_image.shape[0]
                if x_c < 0:
                    x_c = 0
                if y_c > bg_image.shape[1]:
                    y_c = bg_image.shape[1]
                if y_c < 0:
                    x_c = 0
            else:
                x_c = random.randint(0,bg_image.shape[0])
                y_c = random.randint(0,bg_image.shape[1])
            overlay_image(bg_image,random_transformation(leaf_images[key][random.randint(0,len(leaf_images[key])-1)]),mask_image,
                                x_c,y_c,colors,category)
        now = datetime.now()
    
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d%m%Y_%H%M%S")
        #plt.imshow(bg_image)
        filename = 'cb_{0}_{1}.jpg'.format(dt_string,i)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER,filename), bg_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(os.path.join(OUTPUT_FOLDER,filename))
        d1,d2,d3 = mask_image.shape
        all_colors = mask_image.reshape([d1*d2,d3])
        unique_colors = np.unique(all_colors, axis=0)
        new_unique_colors = np.delete(unique_colors, 0,0)

        for color in new_unique_colors[:]:
            mask = np.zeros((mask_image.shape[0],mask_image.shape[1],1),dtype="uint8")
            mask[np.all(mask_image == color, axis=-1)] = 255
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
            #img = cv2.drawContours(leaf_collage, contours, -1, (0,255,75), 2)
            #plt.imshow(img)
            for count in contours:
                epsilon = 0.002 * cv2.arcLength(count, True)
                approximations = cv2.approxPolyDP(count, epsilon, True)
                all_points_x = approximations[:,:,0]
                all_points_y = approximations[:,:,1]
                reshaped_x = all_points_x.flatten()
                reshaped_y = all_points_y.flatten()
                state= ''
                if color[2] == 1:
                    state = 'healthy'
                if color[2] == 2:
                    state = 'withered'
                #img = cv2.drawContours(leaf_collage, [approximations], 0, (0), 3)
                region = {"shape_attributes":{"name":"polygon","all_points_x":reshaped_x.tolist(),"all_points_y":reshaped_y.tolist()},           
                            "region_attributes": {
                            "Type": "Leaf",
                            "State": state,
                            "Sort": "Genovese",
                            "Age": "medium"
                        }}
                regions.append(region)

        filesize = os.path.getsize(os.path.join(OUTPUT_FOLDER,filename))

        annotations[filename + str(filesize)] = {"filename":filename,"size":filesize, "regions":regions,"fileattributes":{}}

    now = datetime.now()

    # dd/mm/YY H:M:S
    #dt_string = now.strftime("%d%m%Y_%H%M%S")
        #plt.imshow(bg_image)
    with open(os.path.join(OUTPUT_FOLDER,'via_region_data.json'), 'w') as f:
        json.dump(annotations, f)