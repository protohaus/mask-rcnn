import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

#creates a new window
root = tk.Tk()

# a label is created
greeting = tk.Label(text="Hello, Tkinter")

label = tk.Label(
    text="Hello, Tkinter",
    foreground="white",  # Set the text color to white
    background="black"  # Set the background color to black
)

button = tk.Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
)

entry = tk.Entry(fg="yellow", bg="blue", width=50)


# one way to put the label in the window
greeting.pack()
label.pack()
button.pack()
entry.pack()

test = entry.get()
print(test)
# let's one select a filedialog
#root.withdraw()
#folder_selected = filedialog.askdirectory()

# needed for interactivity
root.mainloop()