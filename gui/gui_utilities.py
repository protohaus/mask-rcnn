from genericpath import isdir
from tkinter import filedialog
import tkinter as tk
import tkinter.ttk as ttk
import os
import random
from PIL import ImageTk, Image, ImageOps
from gui.AIBatchInference import *
import threading
import webbrowser

class NewWindow(tk.Toplevel):
     
    def __init__(self, master = None,window_title = None,window_label = None,event=None):
         
        super().__init__(master = master)
        self.title(window_title)
        self.event = event
        self.geometry("250x100")
        self.label = ttk.Label(self, text =window_label)
        self.label.pack()
        # progressbar
        pb = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='indeterminate',
            length=280
        )
        pb.pack()
        pb.start()
        self.btn_cancel = ttk.Button(self,text="Cancel",command=self.cancel)
        self.btn_cancel.pack()

    def cancel(self):
        self.label.config(text='Cancelling...')
        self.event.set()

    def monitor_thread(self,thread_to_monitor):
        self.thread = thread_to_monitor
        if thread_to_monitor.is_alive():
            self.after(100,lambda: self.monitor_thread(thread_to_monitor))
        else:
            self.close_window()

    def close_window(self):
        print("Closing Window")
        self.destroy()

def select_file(stringvariable):
    path = filedialog.askopenfilename()
    stringvariable.set(path)

def select_folder(stringvariable):
    path= filedialog.askdirectory()
    stringvariable.set(path)

def show_frame(frames, page_name):
    '''Show a frame for the given page name'''
    frame = frames[page_name]
    frame.tkraise()

def resize_image_height(image,size):
    width,height = image.size
    
    factor = size/height

    return image.resize((int(factor*width),int(factor*height)))

def do_ai_batch_inference(path_model, path_aiweights, path_raw_data):
    event = threading.Event()
    newWindow = NewWindow(master=None,window_title = "AI Batch Progress",window_label="Doing AI Batch Inference. Please wait...",event=event)
    newWindow.update()
    th = threading.Thread(target= batch_inference, args=[path_model,path_aiweights,path_raw_data])
    th.start()
    newWindow.monitor_thread(th)
    return

def open_webbroser(url):
    print(url)
    webbrowser.open(url)

def open_random_image(folder):
    images = []
    image_name = []
    for file in os.listdir(folder):
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            image = Image.open(os.path.join(folder,file))
            image = ImageOps.exif_transpose(image)
            images.append(image)
            image_name.append(file)
    index = random.randint(0,len(images)-1)
    return images[index] , image_name[index]

def load_image(self,height,width, path,resize):
        image , image_name =  open_random_image(path.get())
        maxsize = height
        #image.thumbnail(maxsize, Image.ANTIALIAS)
        if resize:
            image = resize_image_height(image,maxsize)
        self.img = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_container,image=self.img,anchor=tk.NW)
        self.canvas.coords(self.image_container,(width-image.size[0])/2,(height-image.size[1])/2,)
        #canvas.create_image((main_canvas_width-image.size[0])/2,(main_canvas_height-image.size[1])/2, anchor=tk.NW,image=img)
#creates a new window

def count_images(folder):
    count = 0
    if not os.path.exists(folder):
        return 0
    for file in os.listdir(folder):
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            count += 1
        elif os.path.isdir(os.path.join(folder,file)):
            count += count_images(os.path.join(folder,file))
    return count

def load_contours(img_path,json_path,image_name):
    #open json
    with open(json_path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    
    filesize = os.path.getsize(os.path.join(img_path,image_name))   
    
    key = image_name + str(filesize)

    regions = jsonObject[key]["regions"] # list of dicts each dict one region
    contours = []
    for region in regions:
        contour = []
        for i,x in enumerate(region["shape_attributes"]["all_points_x"]):
            point = [x,region["shape_attributes"]["all_points_y"][i]]
            contour.append(point)
        contours.append(np.array(contour))

    return contours

def show_polygons(self,height, width, img_path, json_path):
    image, image_name =  open_random_image(img_path.get())
    maxsize = height
    contours = load_contours(img_path.get(),json_path.get(),image_name)
    pix = np.array(image)
    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(pix, contours, -1, (255, 255, 0), 3)
    image = Image.fromarray(pix)

    image = resize_image_height(image,maxsize)
    self.img = ImageTk.PhotoImage(image)
    self.canvas.itemconfig(self.image_container,image=self.img,anchor=tk.NW)
    self.canvas.coords(self.image_container,(width-image.size[0])/2,(height-image.size[1])/2,)
