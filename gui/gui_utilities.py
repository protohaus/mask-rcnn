from tkinter import filedialog
import tkinter as tk
import tkinter.ttk as ttk
import os
import random
from PIL import ImageTk,Image
from AIBatchInference import *
import time
import threading
import webbrowser

class NewWindow(tk.Toplevel):
     
    def __init__(self, master = None):
         
        super().__init__(master = master)
        self.title("AI Batch Progress")
        self.geometry("250x100")
        label = ttk.Label(self, text ="Doing AI Batch Inference. Please wait...")
        label.pack()
        # progressbar
        pb = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='indeterminate',
            length=280
        )
        pb.pack()
        pb.start()

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
    newWindow = NewWindow()
    newWindow.update()
    th = threading.Thread(target= batch_inference, args=[path_model,path_aiweights,path_raw_data])
    th.start()
    th.join()
    newWindow.close_window()
    return

def open_webbroser(url):
    print(url)
    webbrowser.open(url)

def open_random_image(folder):
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            image = Image.open(os.path.join(folder,file))
            images.append(image)
    return images[random.randint(0,len(images)-1)]

def load_image(self,height,width, path):
        image =  open_random_image(path.get())
        maxsize = height
        #image.thumbnail(maxsize, Image.ANTIALIAS)
        image = resize_image_height(image,maxsize)
        self.img = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_container,image=self.img,anchor=tk.NW)
        self.canvas.coords(self.image_container,(width-image.size[0])/2,(height-image.size[1])/2,)
        #canvas.create_image((main_canvas_width-image.size[0])/2,(main_canvas_height-image.size[1])/2, anchor=tk.NW,image=img)
#creates a new window

def count_images(folder):
    count = 0
    for file in os.listdir(folder.get()):
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            count += 1
    return count


