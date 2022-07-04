from tkinter import *

top = Tk()

class Test(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.x = 0
        self.json = None

        flashCardText = Label(self, text = print('flashcard')).grid(row = 1, column = 2)
        flipButton = Button(self, text = "Yes", command = lambda: print('flip')).grid(row = 2, column = 1)
        nextButton = Button(self, text = "No", command = self._test_output).grid(row = 2, column = 3)
        saveButton = Button(self, text = "Save Data", command = self.save_json).grid(row = 3, column = 1)

        self.grid()

    def load_images(self):
        return

    def save_json(self):
        return

    def _test_output(self):

        if self.x < 2:
            print("increasing x")
            self.x +=1
        else:
            print("x is 2")
            self.master.destroy()

t = Test(top)

top.mainloop()