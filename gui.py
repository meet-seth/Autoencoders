#import torch.nn as nn
import cv2
import torch
import numpy as np
from tkinter import *
from PIL import Image,ImageGrab,ImageTk
from paramtest import Network
import matplotlib.pyplot as plt


def model(x):


    model1 = torch.load('pytorchmodel.pth')
    model1.eval()
    return model1.feedforward(x)


def clear_widget():
    global cv
    i = Image.open('em.jpg')
    lbl_image = ImageTk.PhotoImage(i)
    lbl.configure(image=lbl_image)
    lbl.image = lbl_image
    cv.delete("all")

def activate_event(event):
    global lastx,lasty
    cv.bind('<B1-Motion>',draw_lines)
    lastx,lasty = event.x,event.y

def draw_lines(event):
    global lastx,lasty
    x,y = event.x,event.y
    cv.create_line((lastx,lasty,x,y),width=8,fill='white',
                   capstyle=ROUND,smooth=True,splinesteps=12)
    lastx,lasty=x,y

def Recognise_Digit():
    global image_number

    filename = f'image.png'
    widget = cv

    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()

    ImageGrab.grab().crop((x,y,x1,y1)).save(filename)

    image = cv2.imread(filename,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        img = cv2.resize(th,(28,28),interpolation=cv2.INTER_AREA)
        img = img/255.0
        x1 = np.ones(img.shape)
        img = x1-img
        img = img.reshape(1,1,28,28).astype('float32')
        img = torch.tensor(img)
        pred = model(img)
        final_pred = torch.argmax(pred)
        f = Image.open("{0}.jpg".format(final_pred))
        lbl_image = ImageTk.PhotoImage(f)
        lbl.configure(image=lbl_image)
        lbl.image = lbl_image


root = Tk()
root.resizable(1,1)
root.config(bg="black")
root.title("Handwritten Digit Recognition")

lastx,lasty = None,None
image_number = 0

cv = Canvas(root,width = 280,height = 280,bg = 'black')
cv.grid(row=0,column=0,sticky=W,columnspan=1)
cv.bind('<Button-1>',activate_event)

i = Image.open('em.jpg')
lbl_image = ImageTk.PhotoImage(i)
lbl = Label(root,image=lbl_image)
lbl.grid(row=0,column=1,sticky=NW)
btn_save = Button(text="Recognise Digit",command=Recognise_Digit,fg='black',bg='deepskyblue',width=19)
btn_save.grid(row=2,column=0,sticky=W,pady=1)
btn_clear = Button(text='Clear Widget',command=clear_widget,fg='black',bg='deepskyblue',width=19)
btn_clear.grid(row=2,column=0,sticky=E)

root.mainloop()