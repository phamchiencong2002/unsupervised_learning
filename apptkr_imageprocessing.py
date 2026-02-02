from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os

class ImageProcessing():
    
    def __init__(self):
        # empty image
        self.width = 600
        self.height = 600
        self.image = Image.new("RGB", (self.width, self.height), 'lightgrey')
        
        # start
        self.window = Tk()
        self.window.title("Image processing")
        self.window.resizable(0,0)
        
        icon = Image.open('res/icon.png')
        iconTk = ImageTk.PhotoImage(icon)
        self.window.wm_iconphoto(False, iconTk)
        
        # start
        menu = Menu(self.window)
        self.window.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open", command=self.open_img_file)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.window.destroy)

        # 2 buttons for example
        mdl1_button = Button(self.window, text='Model 1', command=None)
        mdl1_button.grid(row=0, column=0)

        mdl2_button = Button(self.window, text='Model 2', command=None)
        mdl2_button.grid(row=0, column=1)
        
        self.update()
   
    # image file selection
    def open_img_file(self):
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(), 
            title="Select file", 
            filetypes=(("png images", ".png"), ("all files", "*.*")))
        if not filename:
            return
        # read image and update display
        self.image = Image.open(filename)
        self.width, self.height = self.image.size        
        self.update()
    
    # update display    
    def update(self):
        # draw image
        c = Canvas(self.window, bg='white', width=self.width, height=self.height)
        imageTk = ImageTk.PhotoImage(self.image)        
        c.create_image(self.width/2+1, self.height/2+1, image=imageTk)
        # buttons
        c.grid(row=1, columnspan=5)
        self.window.mainloop()

if __name__ == '__main__':
    ImageProcessing()






