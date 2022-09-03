import tkinter as tk
import tkinter.ttk as ttk
import sys
from PIL import ImageTk,Image
from gui.gui_utilities import *
from gui.CreateCollages import create_collages
from gui.description_texts import *
from samples.leafs_collage.leafs_collage import *
import threading
import skimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    ROOT_DIR = os.path.dirname(sys.executable)
elif __file__:
    ROOT_DIR = os.path.dirname(__file__)

#ROOT_DIR = os.path.abspath("/")

via_url = os.path.join(ROOT_DIR,"gui","via-2.0.11","via.html")

menu_row_size = 110
label_pad = 1
grid_pad = 1
main_canvas_width = 700
main_canvas_height = 512

class PageData(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the data page")
        label.pack(side="top", fill="x", pady=label_pad)
        self.canvas = tk.Canvas(self, width = main_canvas_width, height = main_canvas_height, bg = 'gray')      
        self.canvas.pack()
        self.image =  Image.open("gui/raw.jpg")
        maxsize = main_canvas_height
        self.image = resize_image_height(self.image,maxsize)
        self.img = ImageTk.PhotoImage(self.image)   
        self.image_container = self.canvas.create_image((main_canvas_width-self.image.size[0])/2,(main_canvas_height-self.image.size[1])/2, anchor=tk.NW,image=self.img)
        btn_loadIMage = ttk.Button(master=self,text="Show random image",command=lambda: load_image(self,main_canvas_height,main_canvas_width,path_raw_data,True))
        btn_loadIMage.pack()

class PageWeights(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the weights page")
        label.pack(side="top", fill="x", pady=label_pad)
        self.results = []
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
        self.axes = []
        self.opt = tk.OptionMenu(self,weight_layer,*weights_options)
        self.opt.config(width=10, font=('Helvetica', 12))
        self.opt.pack()
        
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)  
        btn_loadIMage = ttk.Button(master=self,text="Inspect Weights",command=lambda: self.inspect_wrapper())
        btn_loadIMage.pack()

    def inspect_wrapper(self):
        self.event = threading.Event()
        self.new_window = NewWindow(master=None,window_title = "Inspecting Weights",window_label="Inspecting selected layers. Please wait...",event=self.event)
        self.new_window.update()
        th = threading.Thread(target= self.inspect_weights)
        th.start()
        self.new_window.monitor_thread(th)

    def inspect_weights(self):
        self.results = show_activation(path_model.get(),path_aiweights.get(),path_input_image.get(),weight_layer.get())
        cols = 2
        titles = None
        titles = titles if titles is not None else [""] * len(self.results)
        rows = len(self.results) // cols + 1
        #self.figure = plt.figure(figsize=(14, 14 * rows // cols))
        #self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
        i = 1
        self.axes = []
        for image in self.results:
            self.axes.append(self.figure.add_subplot(2,2,i))
        #    self.axes.append(self.figure.add_subplot(rows, cols, i))
            #self.axes[i-1].title(title, fontsize=9)
            self.axes[i-1].axis('off')
            self.axes[i-1].imshow(image.astype(np.uint8))
            i += 1
        #plt.show()
        #self.axes[0].imshow(self.results[0].astype(np.uint8))
        self.figure_canvas.draw()


class PageSegmentation(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the segmentation page")
        label.pack(side="top", fill="x", pady=label_pad)
        self.canvas = tk.Canvas(self, width = main_canvas_width, height = main_canvas_height, bg = 'gray')      
        self.canvas.pack()
        image =  Image.open("gui/segmented.png")
        maxsize = main_canvas_height
        image = resize_image_height(image,maxsize)
        self.img = ImageTk.PhotoImage(image)   
        self.image_container = self.canvas.create_image((main_canvas_width-image.size[0])/2,(main_canvas_height-image.size[1])/2, anchor=tk.NW,image=self.img)
        btn_loadIMage = ttk.Button(master=self,text="Show random image",command=lambda: show_polygons(self,main_canvas_height,main_canvas_width,path_raw_data,path_segmentfolder))
        btn_loadIMage.pack()

class PageLeafs(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the leafs page")
        label.pack(side="top", fill="x", pady=label_pad)
        self.canvas = tk.Canvas(self, width = main_canvas_width, height = main_canvas_height, bg = 'gray')      
        self.canvas.pack()
        image =  Image.open("gui/leaf.png")
        maxsize = main_canvas_height
        image = resize_image_height(image,maxsize)
        self.img = ImageTk.PhotoImage(image)   
        self.image_container = self.canvas.create_image((main_canvas_width-image.size[0])/2,(main_canvas_height-image.size[1])/2, anchor=tk.NW,image=self.img)
        btn_loadIMage = ttk.Button(master=self,text="Show random image",command=lambda: load_image(self,main_canvas_height,main_canvas_width,path_leafoutput,False))
        btn_loadIMage.pack()

class PageLeafTinder(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the leaf tinder page")
        #label.pack(side="top", fill="x", pady=label_pad)
        label.grid(row =0, column=1)
        self.canvas_width = main_canvas_width-100
        self.canvas_height = main_canvas_height-100
        self.canvas = tk.Canvas(self, width = self.canvas_width, height = self.canvas_height, bg = 'gray')      
        #self.canvas.pack()
        self.canvas.grid(row =1, column=1)
        self.img_filename = "gui/leaf.png"
        self.image =  Image.open(self.img_filename)
        self.maxsize = main_canvas_height - 100
        self.image = resize_image_height(self.image,self.maxsize)
        self.img = ImageTk.PhotoImage(self.image)   
        self.image_container = self.canvas.create_image((self.canvas_width)/2,(self.canvas_height)/2, anchor=tk.CENTER,image=self.img)
        self.startButton = ttk.Button(self, text = "Start", command = lambda: self.start_tinder())
        self.startButton.grid(row=2, column = 0)
        self.lbl_descr = ttk.Label(self,text = "Leaf category: ")
        self.lbl_descr.grid(row=2, column=1)
        self.lbl_state = ttk.Label(self,text = "")
        self.lbl_state.grid(row=3, column=1)
        self.healthyButton = ttk.Button(self, text = "healthy", command = lambda: self.process_input(True,'healthy'))
        self.healthyButton.grid(row = 4, column = 1)
        self.witheredButton = ttk.Button(self, text = "withered", command = lambda: self.process_input(True,'withered'))
        self.witheredButton.grid(row = 5, column = 1)
        self.noButton = ttk.Button(self, text = "Discard", command = lambda: self.process_input(False,''))
        self.noButton.grid(row = 6, column = 1)
        self.saveButton = ttk.Button(self, text = "Save Data", command = self.save_json)
        self.saveButton.grid(row = 7, column = 0)
        self.healthyButton['state'] = tk.DISABLED
        self.witheredButton['state'] = tk.DISABLED
        self.noButton['state'] = tk.DISABLED
        self.saveButton['state'] = tk.DISABLED

    def export_leafs(self):
        healthypath =os.path.join( path_leafoutput.get(),'healthy')
        witheredpath =os.path.join( path_leafoutput.get(),'withered')
        if not os.path.exists(healthypath):
            os.makedirs(healthypath)
        if not os.path.exists(witheredpath):
            os.makedirs(witheredpath)

        for i, leaf in enumerate(self.leafs):
            key, filename, index = self.convert_index_to_json(self.leafs[i]["index"])
            if self.jsonObject[key]["regions"][index]["region_attributes"]["leaftinder"] == "yes":
                if self.jsonObject[key]["regions"][index]["region_attributes"]["State"] == "healthy":
                #cv2.imwrite(os.path.join(path_leafoutput.get(),'IMG_{0}_ROI_mask_{1}.png'.format(filename,index)), leaf["image"])
                    leaf["image"].save(os.path.join(healthypath,'IMG_{0}_ROI_mask_{1}.png'.format(filename,index)))
                if self.jsonObject[key]["regions"][index]["region_attributes"]["State"] == "withered":
                #cv2.imwrite(os.path.join(path_leafoutput.get(),'IMG_{0}_ROI_mask_{1}.png'.format(filename,index)), leaf["image"])
                    leaf["image"].save(os.path.join(witheredpath,'IMG_{0}_ROI_mask_{1}.png'.format(filename,index)))

    def save_json(self):
        # save json
        self.startButton['state'] = tk.NORMAL
        self.healthyButton['state'] = tk.DISABLED
        self.witheredButton['state'] = tk.DISABLED
        self.noButton['state'] = tk.DISABLED
        self.saveButton['state'] = tk.DISABLED
        with open(path_segmentfolder.get(), 'w') as f:
            json.dump(self.jsonObject, f)
        self.export_leafs()
        return

    def process_input(self,decision,category):
        key, filename, index = self.convert_index_to_json(self.leafs[self.index]["index"])
        #self.jsonObject[key]["regions"][index]["region_attributes"]["leaftinder"]
        if decision == True:
            self.jsonObject[key]["regions"][index]["region_attributes"]["leaftinder"] = "yes"
            self.jsonObject[key]["regions"][index]["region_attributes"]["State"] = category
            print('yes')
        else:
            print('no')
            self.jsonObject[key]["regions"][index]["region_attributes"]["leaftinder"] = "no"
        self.show_next_image()
        return
    
    def show_next_image(self):
        self.index += 1
        if self.index >= len(self.leafs):
            self.index = 0
            self.save_json()
        # show next undecided image
        self.image = self.leafs[self.index]["image"]
        key, filename, index = self.convert_index_to_json(self.leafs[self.index]["index"])
        self.lbl_state.config(text=self.jsonObject[key]["regions"][index]["region_attributes"]["State"])
        #image.thumbnail(maxsize, Image.ANTIALIAS)
        #image = resize_image_height(image,self.maxsize)
        self.img = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_container,image=self.img,anchor=tk.CENTER)
        self.canvas.coords(self.image_container,(self.canvas_width)/2,(self.canvas_height)/2)
        return

    def start_tinder(self):
        # Check if everything is allright
        self.index = -1
        self.leafs = []
        #open json
        with open(path_segmentfolder.get()) as jsonFile:
            self.jsonObject = json.load(jsonFile)
            jsonFile.close()
        
        self.load_leafs(path_raw_data.get())

        self.show_next_image()
        #filesize = os.path.getsize(os.path.join(img_path,image_name))   
    
        #key = image_name + str(filesize)
        # load images
        # check output folder
        self.startButton['state'] = tk.DISABLED
        self.healthyButton['state'] = tk.NORMAL
        self.witheredButton['state'] = tk.NORMAL
        self.noButton['state'] = tk.NORMAL
        self.saveButton['state'] = tk.NORMAL
        return

    def load_leafs(self,folder):
        # self.images = []
        # self.image_names = []
        # for file in os.listdir(folder):
        #     if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
        #         image = Image.open(os.path.join(folder,file))
        #         self.images.append(image)
        #         self.image_names.append(file)
        annotations = list(self.jsonObject.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        lastIndex = 0
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            img = cv2.imread(os.path.join(folder,a['filename']),flags=cv2.IMREAD_ANYCOLOR)
            if img is None:
                print("img is NoneType")
                continue
            height, width, channels = img.shape
            #print(os.path.join(LEAF_FOLDER,image))
            #print(leaf_image.shape)
            new_width = int(img.shape[1] * 33 / 100)
            new_height = int(img.shape[0] * 33 / 100)
            dim = (new_width, new_height)

            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR_EXACT)

            masks = np.zeros([new_height, new_width, len(polygons)],dtype=np.uint8)

            for i, p in enumerate(polygons):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

                #fixes index out of bounds as described by https://github.com/matterport/Mask_RCNN/issues/636

                mask = np.zeros([height,width,1])
                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1
                mask[rr, cc] = 1
                masks[:, :, i] = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)

            #%%
            boxes = np.zeros([masks.shape[-1], 4], dtype=np.int32)

            for i in range(masks.shape[-1]):
                m = masks[:, :, i]
                # Bounding box.
                horizontal_indicies = np.where(np.any(m, axis=0))[0]
                vertical_indicies = np.where(np.any(m, axis=1))[0]
                if horizontal_indicies.shape[0]:
                    x1, x2 = horizontal_indicies[[0, -1]]
                    y1, y2 = vertical_indicies[[0, -1]]
                    # x2 and y2 should not be part of the box. Increment by 1.
                    x2 += 1
                    y2 += 1
                else:
                    # No mask for this instance. Might happen due to
                    # resizing or cropping. Set bbox to zeros
                    x1, x2, y1, y2 = 0, 0, 0, 0
                boxes[i] = np.array([y1, y2,x1, x2])

            #print(boxes)

            for index, box in enumerate(boxes):
                ROI = resized[box[0]:box[1],box[2]:box[3]]
                if ROI.shape[0] == 0:
                    print("ROI is NoneType")
                    continue
                # First create the image with alpha channel
                rgba = cv2.cvtColor(ROI, cv2.COLOR_RGB2RGBA)
                # Then assign the mask to the last channel of the image
                mask = masks[box[0]:box[1],box[2]:box[3],index]*255
                rgba[:, :, 3] = mask
                self.leafs.append({"index": index + lastIndex, "image":Image.fromarray(rgba)})
                #cv2.imwrite('./ResultingLeafimages/IMG_{0}_ROI_mask{1}.png'.format(j,index), mask)
                #cv2.imwrite('./ResultingLeafimages/IMG_{0}_ROI_mask_{1}.png'.format(a['filename'],index), rgba)
            lastIndex += len(boxes)

    def convert_index_to_json(self,index):
        for key, image in self.jsonObject.items():
            #key = image.keys()[i]
            filename = image["filename"]
            num = len(image["regions"])
            if index - num < 0:
                break
            else:
                index = index - num
        return key, filename, index

class PageCollage(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the collages page")
        label.pack(side="top", fill="x", pady=label_pad)
        self.canvas = tk.Canvas(self, width = main_canvas_width, height = main_canvas_height, bg = 'gray')      
        self.canvas.pack()
        self.image =  Image.open("gui/collages.png")
        maxsize = main_canvas_height
        self.image = resize_image_height(self.image,maxsize)
        self.img = ImageTk.PhotoImage(self.image)
        self.image_container = self.canvas.create_image((main_canvas_width-self.image.size[0])/2,(main_canvas_height-self.image.size[1])/2, anchor=tk.NW,image=self.img)
        btn_loadIMage = ttk.Button(master=self,text="Show random image",command=lambda: load_image(self,main_canvas_height,main_canvas_width,path_collagefolder,True))
        btn_loadIMage.pack()

class PageBackgrounds(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the backgrounds page")
        label.pack(side="top", fill="x", pady=label_pad)
        self.canvas = tk.Canvas(self, width = main_canvas_width, height = main_canvas_height, bg = 'gray')      
        self.canvas.pack()
        self.image =  Image.open("gui/collages.png")
        maxsize = main_canvas_height
        self.image = resize_image_height(self.image,maxsize)
        self.img = ImageTk.PhotoImage(self.image)
        self.image_container = self.canvas.create_image((main_canvas_width-self.image.size[0])/2,(main_canvas_height-self.image.size[1])/2, anchor=tk.NW,image=self.img)
        btn_loadIMage = ttk.Button(master=self,text="Show random image",command=lambda: load_image(self,main_canvas_height,main_canvas_width,path_backgroundfolder,True))
        btn_loadIMage.pack()

class PageTrain(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the train page")
        label.grid(row=0,column=0, sticky="ew")
        #self.canvas = tk.Canvas(self, width = main_canvas_width, height = main_canvas_height, bg = 'gray')      
        #self.canvas.pack()
        #self.image =  Image.open("gui/cnn.png")
        #maxsize = main_canvas_height
        #self.image = resize_image_height(self.image,maxsize)
        #self.img = ImageTk.PhotoImage(self.image)   
        #self.image_container = self.canvas.create_image((main_canvas_width-self.image.size[0])/2,(main_canvas_height-self.image.size[1])/2, anchor=tk.NW,image=self.img)
        self.lbl_layers = ttk.Label(master=self, text = "Layers to train: ")
        self.lbl_layers.grid(row=1,column=0,sticky="w")
        self.opt = tk.OptionMenu(self,train_layers,*train_options)
        self.opt.config(width=10, font=('Helvetica', 12))
        self.opt.grid(row=1,column=1, sticky="ew")
        self.lbl_num_epochs = ttk.Label(self, text="Number of Epochs")
        self.lbl_num_epochs.grid(row=2,column=0,sticky="w")
        self.ent_num_epochs = ttk.Entry(master=self,textvariable=number_of_epochs)
        self.ent_num_epochs.grid(row=2,column=1,sticky="e")
        self.lbl_learning_rate = ttk.Label(self, text="Learning Rate")
        self.lbl_learning_rate.grid(row=3,column=0,sticky="w")
        self.ent_learning_rate = ttk.Entry(master=self,textvariable=learning_rate)
        self.ent_learning_rate.grid(row=3,column=1,sticky="e")
        btn_startTraining = ttk.Button(master=self,text="Start Training",
            command=lambda: self.train_wrapper())
        btn_startTraining.grid(row=10,column=3, sticky="e")

    def train_wrapper(self):
        self.event = threading.Event()
        self.new_window = NewWindow(master=None,window_title = "Training",window_label="Started training. This will take a long time. Please wait...",event=self.event)
        self.new_window.update()
        th = threading.Thread(target= train_from_import, args =[path_trainfolder.get(),path_aiweights.get(),path_collagefolder.get(),
                train_layers.get(),int(number_of_epochs.get()),float(learning_rate.get()),self.event])
        th.start()
        self.new_window.monitor_thread(th)

class PageTest(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Inspector: This is the test page")
        label.pack(side="top", fill="x", pady=label_pad)
        #self.canvas = tk.Canvas(self, width = main_canvas_width, height = main_canvas_height, bg = 'gray')      
        #self.canvas.pack()
        #self.image =  Image.open("gui/test.png")
        #maxsize = main_canvas_height
        self.results = {}
        #self.image = resize_image_height(self.image,maxsize)
        #self.img = ImageTk.PhotoImage(self.image)   
        #self.image_container = self.canvas.create_image((main_canvas_width-self.image.size[0])/2,(main_canvas_height-self.image.size[1])/2, anchor=tk.NW,image=self.img)
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
        self.axes = self.figure.add_subplot()
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        btn_loadIMage = ttk.Button(master=self,text="Validate Testset",command=lambda: self.test_wrapper())
        btn_loadIMage.pack()

    def test_wrapper(self):
        self.event = threading.Event()
        self.new_window = NewWindow(master=None,window_title = "Validation",window_label="Started validating. This will might a long time. Please wait...",event = self.event)
        self.new_window.update()
        th = threading.Thread(target= self.validate)
        th.start()
        self.new_window.monitor_thread(th)

    def validate(self):
        batch_validation(path_model.get(),path_aiweights.get(),path_testfolder.get(),self.results)
        print("Done validating")
        self.axes.plot(self.results["precisions"],self.results["recalls"])
        self.figure_canvas.draw()

class PageInfoData(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Inspector: This is the data info page")
        self.label.pack(side="top", fill="x", pady=label_pad)
        self.label_info = ttk.Label(self, text=raw_data_text)
        self.label_info.pack(side="top", fill="x", pady=label_pad)

class PageInfoWeights(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Inspector: This is the weights info page")
        self.label.pack(side="top", fill="x", pady=label_pad)
        self.label_info = ttk.Label(self, text=ai_weights_text)
        self.label_info.pack(side="top", fill="x", pady=label_pad)

class PageInfoSegmentation(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Inspector: This is the segmentation info page")
        self.label.pack(side="top", fill="x", pady=label_pad)
        self.label_info = ttk.Label(self, text=segmentation_text)
        self.label_info.pack(side="top", fill="x", pady=label_pad)

class PageInfoLeafTinder(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Inspector: This is the leaf tinder info page")
        self.label.pack(side="top", fill="x", pady=label_pad)
        self.label_info = ttk.Label(self, text=leaf_tinder_text)
        self.label_info.pack(side="top", fill="x", pady=label_pad)

class PageInfoCollage(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Inspector: This is the collages info page")
        self.label.pack(side="top", fill="x", pady=label_pad)
        self.label_info = ttk.Label(self, text=leaf_collages_text)
        self.label_info.pack(side="top", fill="x", pady=label_pad)

class PageInfoTrain(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Inspector: This is the train info page")
        self.label.pack(side="top", fill="x", pady=label_pad)
        self.label_info = ttk.Label(self, text=training_text)
        self.label_info.pack(side="top", fill="x", pady=label_pad)

class PageInfoTest(tk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.label = ttk.Label(self, text="Inspector: This is the test infopage")
        self.label.pack(side="top", fill="x", pady=label_pad)
        self.label_info = ttk.Label(self, text=test_text)
        self.label_info.pack(side="top", fill="x", pady=label_pad)

class PageMenuRawData(ttk.Frame):
    def __init__(self, parent):
        self.frm_data= ttk.Frame(master=parent,relief=tk.RAISED,border=5)
        self.frm_data.grid(row=0,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
        self.frm_data.columnconfigure(0,minsize=200)
        self.frm_data.columnconfigure(1,minsize=40)
        #self.img = ImageTk.PhotoImage(Image.open("gui/rawdata.png"))
        self.lbl_data = ttk.Label(master=self.frm_data,text="Raw Data")
        self.lbl_data.grid(row=0,column=0, sticky = "w")
        self.ent_datafolder = ttk.Entry(master=self.frm_data,textvariable=path_raw_data)
        #ent_datafolder.insert(-1,textvariable=path_raw_data)
        self.ent_datafolder.grid(row=1,column=0, sticky = "ew")
        self.btn_choosedatafolder = ttk.Button(master=self.frm_data,text="Browse",command=lambda: select_folder(path_raw_data))
        self.btn_choosedatafolder.grid(row=1,column=1,sticky="e")
        self.btn_showdata = ttk.Button(master=self.frm_data,text="Show Data",command=lambda: [show_frame(frames,"PageData"),show_frame(frames,"PageInfoData")])
        self.btn_showdata.grid(row=1,column=2,sticky="e")
        self.lbl_imagenumber = ttk.Label(master=self.frm_data,text="0 images")
        self.lbl_imagenumber.grid(row=2,column=0,sticky = "w")
        self.update_count()

    def update_count(self):
        count = count_images(path_raw_data.get()) 
        self.lbl_imagenumber.configure(text = str(count) + " images")
        self.lbl_imagenumber.after(1000, self.update_count)

class PageMenuLeafTinder(ttk.Frame):
    def __init__(self,parent):
        self.frm_leaftinder= ttk.Frame(master=parent,relief=tk.RAISED,border=5)
        self.frm_leaftinder.grid(row=3,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
        self.frm_leaftinder.columnconfigure(0,minsize=200)
        self.frm_leaftinder.columnconfigure(1,minsize=40)
        self.lbl_leaftinder = ttk.Label(master=self.frm_leaftinder,text="Leaf Tinder")
        self.lbl_leaftinder.grid(row=0,column=0, sticky = "w")
        self.ent_leaffolder = ttk.Entry(master=self.frm_leaftinder,textvariable=path_leafoutput)
        #ent_leaffolder.insert(-1,'C:/Outputfolder')
        self.ent_leaffolder.grid(row=1,column=0, sticky = "ew")
        self.btn_chooseleaffolder = ttk.Button(master=self.frm_leaftinder,text="Browse",command=lambda: select_folder(path_leafoutput))
        self.btn_chooseleaffolder.grid(row=1,column=1,sticky="e")
        self.btn_starttinder = ttk.Button(master=self.frm_leaftinder,text="Start Leaf Tinder",command=lambda: [show_frame(frames,"PageLeafTinder"),show_frame(frames,"PageInfoLeafTinder")])
        self.btn_starttinder.grid(row=1,column=2,sticky="w")
        self.btn_showleafs = ttk.Button(master=self.frm_leaftinder,text="Show Leafs",command=lambda: [show_frame(frames,"PageLeafs"),show_frame(frames,"PageInfoLeafTinder")])
        self.btn_showleafs.grid(row=3,column=0,sticky="e")
        self.lbl_imagenumber = ttk.Label(master=self.frm_leaftinder,text="0 images")
        self.lbl_imagenumber.grid(row=2,column=0,sticky = "w")
        self.update_count()

    def update_count(self):
        count = count_images(path_leafoutput.get())
        self.lbl_imagenumber.configure(text = str(count) + " images")
        self.lbl_imagenumber.after(1000, self.update_count)

class PageMenuCollages(ttk.Frame):
    def __init__(self,parent):
        self.frm_collages= ttk.Frame(master=frm_menu,relief=tk.RAISED,border=5)
        self.frm_collages.grid(row=4,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
        self.frm_collages.columnconfigure(0,minsize=200)
        self.frm_collages.columnconfigure(1,minsize=40)
        self.lbl_collages = ttk.Label(master=self.frm_collages,text="Leaf Collages")
        self.lbl_collages.grid(row=0,column=0, sticky = "w")

        self.ent_backgroundfolder = ttk.Entry(master=self.frm_collages,textvariable=path_backgroundfolder)
        self.ent_backgroundfolder.grid(row=1,column=0, sticky = "ew")

        self.btn_choosebackgroundfolder = ttk.Button(master=self.frm_collages,text="Browse",command=lambda: select_folder(path_backgroundfolder))
        self.btn_choosebackgroundfolder.grid(row=1,column=1,sticky="e")
        self.btn_showbackgrounds = ttk.Button(master=self.frm_collages,text="Show Backgrounds",command=lambda: [show_frame(frames,"PageBackgrounds"),show_frame(frames,"PageInfoCollage")])
        self.btn_showbackgrounds.grid(row=1,column=2,sticky="e")
        
        self.lbl_bgnumber = ttk.Label(master=self.frm_collages,text="0 images")
        self.lbl_bgnumber.grid(row=2,column=0,sticky = "w")  

        self.ent_collagesfolder = ttk.Entry(master=self.frm_collages,textvariable=path_collagefolder)
        #ent_collagesfolder.insert(-1,'C:/CollageFolder')
        self.ent_collagesfolder.grid(row=3,column=0, sticky = "ew")
        self.btn_choosecollagesfolder = ttk.Button(master=self.frm_collages,text="Browse",command=lambda: select_folder(path_collagefolder))
        self.btn_choosecollagesfolder.grid(row=3,column=1,sticky="e")
        self.btn_showcollages = ttk.Button(master=self.frm_collages,text="Show Collages",command=lambda: [show_frame(frames,"PageCollage"),show_frame(frames,"PageInfoCollage")])
        self.btn_showcollages.grid(row=3,column=2,sticky="e")

        self.lbl_imagenumber = ttk.Label(master=self.frm_collages,text="0 images")
        self.lbl_imagenumber.grid(row=4,column=0,sticky = "w")
        self.lbl_numberofsamples = ttk.Label(master=self.frm_collages, text = "Input number of collages to create")
        self.lbl_numberofsamples.grid(row=5,column=0,sticky="w")
        self.ent_numberofsamples = ttk.Entry(master=self.frm_collages,textvariable=numberofleafs)
        self.ent_numberofsamples.grid(row=5,column=1,sticky="ew")
        self.btn_createcollages = ttk.Button(master=self.frm_collages,text="Create Collages",command=lambda:self.collage_wrapper())
        self.btn_createcollages.grid(row=5,column=2,sticky="e")

        self.update_count()
        #self.update_count()

    def collage_wrapper(self):
        self.event = threading.Event()
        self.new_window = NewWindow(master=None,window_title = "Collages",window_label="Creating leaf train collages. Please wait...",event = self.event)
        self.new_window.update()
        th = threading.Thread(target= create_collages, args =[path_leafoutput.get(),path_backgroundfolder.get(),os.path.join(path_collagefolder.get(),'train'), int(self.ent_numberofsamples.get())])
        th.start()
        self.new_window.monitor_thread(th)
        self.event2 = threading.Event()
        self.new_window = NewWindow(master=None,window_title = "Collages",window_label="Creating leaf val collages. Please wait...",event=self.event2)
        self.new_window.update()
        th2 = threading.Thread(target= create_collages, args =[path_leafoutput.get(),path_backgroundfolder.get(),os.path.join(path_collagefolder.get(),'val'), int(int(self.ent_numberofsamples.get())/4)])
        th2.start()
        self.new_window.monitor_thread(th2)

    def update_count(self):
        count = count_images(path_collagefolder.get())
        self.lbl_imagenumber.configure(text = str(count) + " images")
        count = count_images(path_backgroundfolder.get())
        self.lbl_bgnumber.configure(text = str(count) + " images")
        self.lbl_imagenumber.after(1000, self.update_count)

window = tk.Tk()
window.title("OFAI GUI")

window.columnconfigure(0, minsize=260)
window.columnconfigure(1, minsize=1270)
window.rowconfigure([0, 0], minsize=780)

path_raw_data = tk.StringVar(value = "D:/Protohaus/Basil_Database/Basilikum/Basilikum/genovese/erwachsen/gesund/Testset")
path_aiweights = tk.StringVar(value = "D:/Protohaus/GitHub/mask-rcnn/mask_rcnn_leafscollage_2classes.h5")
path_input_image = tk.StringVar(value = "D:/Protohaus/Basil_Database/Basilikum/Basilikum/genovese/erwachsen/gesund/Testset/20210721_100816.jpg")
path_model = tk.StringVar(value = "D:/Protohaus/GitHub/mask-rcnn/")
path_leafoutput = tk.StringVar(value = "D:/Protohaus/LeafOutput")
path_collagefolder = tk.StringVar(value = "D:/Protohaus/CollageFolder")
path_backgroundfolder = tk.StringVar(value = "D:/Protohaus/BackgroundFolder")
path_trainfolder = tk.StringVar(value = "D:/Protohaus/TrainFolder")
path_testfolder = tk.StringVar(value = "D:/Protohaus/Basil_Database/test")
path_segmentfolder = tk.StringVar(value = "D:/Protohaus/Segmented")

weights_options = ["activation","activation_1","activation_2","res2a_out","activation_3","activation_4",
                "res2b_out","activation_5","activation_6","res2c_out","activation_7","activation_8",
                "res3a_out","activation_9","activation_10","res3b_out","activation_11","activation_12",
                "res3c_out","activation_13"  ,"activation_14","res3d_out" ,"activation_15", "activation_16",
                "res4a_out","activation_17","activation_18","res4b_out","activation_19","activation_20",
                "res4c_out","activation_21","activation_22","res4d_out","activation_23","activation_24",
                "res4e_out","activation_25","activation_26","res4f_out","activation_27","activation_28",
                "res4g_out","activation_29","activation_30","res4h_out","activation_31","activation_32",
                "res4i_out","activation_33","activation_34","res4j_out","activation_35","activation_36",
                "res4k_out","activation_37","activation_38","res4l_out","activation_39","activation_40",
                "res4m_out","activation_41","activation_42","res4n_out","activation_43","activation_44",
                "res4o_out","activation_45","activation_46","res4p_out","activation_47","activation_48",
                "res4q_out","activation_49","activation_50","res4r_out","activation_51","activation_52",
                "res4s_out","activation_53","activation_54","res4t_out","activation_55","activation_56",
                "res4u_out","activation_57","activation_58","res4v_out","activation_59","activation_60",
                "res4w_out","activation_61","activation_62","res5a_out","activation_63","activation_64",
                "res5b_out","activation_65","activation_66","res5c_out","activation_67","activation_68",
                "activation_70","activation_71","activation_72","activation_73"]
weight_layer = tk.StringVar()
weight_layer.set(weights_options[0])

train_options = ["3+","4+","5+","heads","all"]
train_layers = tk.StringVar()
train_layers.set(train_options[0])
number_of_epochs = tk.StringVar(value="100")
learning_rate = tk.StringVar(value="0.002")

numberofleafs = tk.StringVar(value="100")

frm_menu = ttk.Frame(master=window,relief=tk.RAISED,border=5)
frm_menu.grid(row=0,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
#lbl_menu = tk.Label(master=frm_menu,text="Menu Pane")
#lbl_menu.pack()
for i in range(8):
    frm_menu.rowconfigure([0, i], minsize=menu_row_size)

datapage = PageMenuRawData(frm_menu)

frm_aiweights= ttk.Frame(master=frm_menu,relief=tk.RAISED,border=5)
frm_aiweights.grid(row=1,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
frm_aiweights.columnconfigure(0,minsize=200)
frm_aiweights.columnconfigure(1,minsize=40)
lbl_aiweights = ttk.Label(master=frm_aiweights,text="AI Weights")
lbl_aiweights.grid(row=0,column=0, sticky = "w")
ent_weightsfile = ttk.Entry(master=frm_aiweights,textvariable=path_aiweights)
#ent_weightsfile.insert(-1,'C:/Weights')
ent_weightsfile.grid(row=1,column=0, sticky = "ew")
btn_chooseweightsfolder = ttk.Button(master=frm_aiweights,text="Browse",command=lambda: select_file(path_aiweights))
btn_chooseweightsfolder.grid(row=1,column=1,sticky="e")
ent_weightsinputfile = ttk.Entry(master=frm_aiweights,textvariable=path_input_image)
#ent_weightsfile.insert(-1,'C:/Weights')
ent_weightsinputfile.grid(row=2,column=0, sticky = "ew")
btn_chooseweightsinputfile = ttk.Button(master=frm_aiweights,text="Browse",command=lambda: select_file(path_input_image))
btn_chooseweightsinputfile.grid(row=2,column=1,sticky="e")
btn_showweights = ttk.Button(master=frm_aiweights,text="Show Weights",command=lambda: [show_frame(frames,"PageWeights"),show_frame(frames,"PageInfoWeights")])
btn_showweights.grid(row=3,column=1,sticky="e")

frm_segmentation= ttk.Frame(master=frm_menu,relief=tk.RAISED,border=5)
frm_segmentation.grid(row=2,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
frm_segmentation.columnconfigure(0,minsize=200)
frm_segmentation.columnconfigure(1,minsize=40)
lbl_segmentation = ttk.Label(master=frm_segmentation,text="Segmentation")
lbl_segmentation.grid(row=0,column=0, sticky = "w")
#ent_modelfolder = ttk.Entry(master=frm_segmentation,textvariable=path_model)
#ent_weightsfile.insert(-1,'C:/Weights')
#ent_modelfolder.grid(row=0,column=1, sticky = "e")
#btn_choosemodelfolder = ttk.Button(master=frm_segmentation,text="Browse",command=lambda: select_folder(path_model))
#btn_choosemodelfolder.grid(row=1,column=2,sticky="e")
btn_aisegmentation = ttk.Button(master=frm_segmentation,text="AI Segmentation",command=lambda: [do_ai_batch_inference(path_model.get(),path_aiweights.get(),path_raw_data.get())])
btn_aisegmentation.grid(row=1,column=0,sticky="w")
btn_manualsegmentation = ttk.Button(master=frm_segmentation,text="Manual Segmentation",command=lambda: open_webbroser(via_url))
btn_manualsegmentation.grid(row=1,column=1)
#btn_clearsegmentation = ttk.Button(master=frm_segmentation,text="Clear Data")
#btn_clearsegmentation.grid(row=2,column=0,sticky="w")
ent_json = ttk.Entry(master=frm_segmentation,textvariable=path_segmentfolder)
#ent_weightsfile.insert(-1,'C:/Weights')
ent_json.grid(row=3,column=1, sticky = "e")
btn_choosejson = ttk.Button(master=frm_segmentation,text="Browse",command=lambda: select_file(path_segmentfolder))
btn_choosejson.grid(row=3,column=2,sticky="e")
btn_showsegmentation = ttk.Button(master=frm_segmentation,text="Show Segmentation",command=lambda: [show_frame(frames,"PageSegmentation"),show_frame(frames,"PageInfoSegmentation")])
btn_showsegmentation.grid(row=4,column=0,sticky="e")

leaftinderpage = PageMenuLeafTinder(frm_menu)

collages_page = PageMenuCollages(frm_menu)

frm_train= ttk.Frame(master=frm_menu,relief=tk.RAISED,border=5)
frm_train.grid(row=5,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
frm_train.columnconfigure(0,minsize=200)
frm_train.columnconfigure(1,minsize=40)
lbl_train = ttk.Label(master=frm_train,text="Training")
lbl_train.grid(row=0,column=0, sticky = "w")
ent_trainfolder = ttk.Entry(master=frm_train,textvariable=path_trainfolder)
#ent_trainfolder.insert(-1,'C:/CollageFolder')
ent_trainfolder.grid(row=1,column=0, sticky = "ew")
btn_choosetrainfolder = ttk.Button(master=frm_train,text="Browse",command=lambda: select_folder(path_trainfolder))
btn_choosetrainfolder.grid(row=1,column=1,sticky="e")
btn_train = ttk.Button(master=frm_train,text="Configure Training",command=lambda: [show_frame(frames,"PageTrain"),show_frame(frames,"PageInfoTrain")])
btn_train.grid(row=1,column=2,sticky="w")

frm_test= ttk.Frame(master=frm_menu,relief=tk.RAISED,border=5)
frm_test.grid(row=6,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
frm_test.columnconfigure(0,minsize=200)
frm_test.columnconfigure(1,minsize=40)
lbl_test = ttk.Label(master=frm_test,text="Test Set Performance")
lbl_test.grid(row=0,column=0, sticky = "w")
ent_testfolder = ttk.Entry(master=frm_test,textvariable=path_testfolder)
#ent_testfolder.insert(-1,'C:/CollageFolder')
ent_testfolder.grid(row=1,column=0, sticky = "ew")
btn_choosetestfolder = ttk.Button(master=frm_test,text="Browse",command=lambda: select_folder(path_testfolder))
btn_choosetestfolder.grid(row=1,column=1,sticky="e")
btn_test = ttk.Button(master=frm_test,text="Test AI",command=lambda: [show_frame(frames,"PageTest"),show_frame(frames,"PageInfoTest")])
btn_test.grid(row=1,column=2,sticky="w")

frm_main = ttk.Frame(master=window,relief=tk.RAISED,border=5)
frm_main.grid(row=0,column=1, padx=1, pady=1,sticky="nsew")
#lbl_main = tk.Label(master=frm_main, text="Main")
#lbl_main.pack()

frm_main.rowconfigure([0, 1], minsize=585)
frm_main.rowconfigure([1, 1], minsize=185)
frm_main.columnconfigure(0, minsize=1060)

frm_main_up = ttk.Frame(master=frm_main,relief=tk.RAISED,border=5)
frm_main_up.grid(row=0,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
frm_main_up.columnconfigure(0, minsize=300)
frm_main_up.columnconfigure(1, minsize=760)
frm_main_up.rowconfigure([0,0],minsize=575)

frm_info = ttk.Frame(master=frm_main_up,relief=tk.RAISED,border=5)
frm_info.grid(row=0,column=0,padx=grid_pad, pady=grid_pad, sticky="nsew")
#lbl_info = tk.Label(master=frm_info, text = "Info Window")
#lbl_info.pack()

frm_inspector = ttk.Frame(master=frm_main_up,relief=tk.RAISED,border=5)
frm_inspector.grid(row=0,column=1,padx=grid_pad, pady=grid_pad, sticky="nsew")
#lbl_inspector = tk.Label(master=frm_inspector, text="Inspector")
#lbl_inspector.pack()

frm_main_down = ttk.Frame(master=frm_main,relief=tk.RAISED,border=5)
frm_main_down.grid(row=1,column=0, padx=grid_pad, pady=grid_pad, sticky="nsew")
frm_main_down.columnconfigure(0, minsize=760)
frm_main_down.columnconfigure(1, minsize=300)
frm_main_down.rowconfigure([0,0],minsize=175)

frm_pipeline = ttk.Frame(master=frm_main_down,relief=tk.RAISED,border=5)
frm_pipeline.grid(row=0,column=0,padx=grid_pad, pady=grid_pad, sticky="nsew")
lbl_pipeline = ttk.Label(master=frm_pipeline, text="Pipeline")
lbl_pipeline.pack()
canvas_pipeline = tk.Canvas(frm_pipeline,height=125,width=385)
canvas_pipeline.pack()
pipeline_img = ImageTk.PhotoImage(Image.open("gui/Pipeline.png").resize((385,125), Image.ANTIALIAS))
canvas_pipeline.create_image(0,0, anchor=tk.NW,image=pipeline_img)

frm_logo = ttk.Frame(master=frm_main_down,relief=tk.RAISED,border=5)
frm_logo.grid(row=0,column=1,padx=grid_pad, pady=grid_pad, sticky="nsew")
#lbl_logo = tk.Label(master=frm_logo, text="Logo")
#lbl_logo.pack()
canvas_logo = tk.Canvas(frm_logo, width = 300, height = 300, bg = 'white')      
canvas_logo.pack()      
img =  ImageTk.PhotoImage(Image.open("gui/ofai.png"))        
canvas_logo.create_image(0,0, anchor=tk.NW,image=img)   

frames = {}
for F in (PageData, PageWeights, PageSegmentation, PageLeafTinder,PageLeafs, PageCollage,
            PageBackgrounds, PageTrain, PageTest):
    page_name = F.__name__
    frame = F(parent=frm_inspector)
    frames[page_name] = frame

    # put all of the pages in the same location;
    # the one on the top of the stacking order
    # will be the one that is visible.
    frame.grid(row=0, column=0, sticky="nsew")

for F in (PageInfoData, PageInfoWeights, PageInfoSegmentation, PageInfoLeafTinder, PageInfoCollage, 
        PageInfoTrain, PageInfoTest):
    page_name = F.__name__
    frame = F(parent=frm_info)
    frames[page_name] = frame

    # put all of the pages in the same location;
    # the one on the top of the stacking order
    # will be the one that is visible.
    frame.grid(row=0, column=0, sticky="nsew")

show_frame(frames,"PageData")
show_frame(frames,"PageInfoData")

# needed for interactivity
window.mainloop()