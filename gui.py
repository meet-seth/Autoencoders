#import torch.nn as nn
import cv2
import tensorflow as tf
import numpy as np
from tkinter import *
from PIL import Image,ImageDraw,ImageTk
import matplotlib.pyplot as plt
from io import BytesIO


def model(x):

    global image_number,encoder,decoder
      
    
    compressed = encoder.serve(x)
    regenerated = decoder.serve(compressed)
    
    with open(f"compressed_imgs/{image_number}.cmp",'wb') as f:
        f.write(compressed.numpy().tobytes())
        f.close()
    
    image_number+=1
    return regenerated.numpy()[0]*255.


def clear_widget():
    global cv, new_img,draw
    new_img = Image.new("L",(280,280))
    draw = ImageDraw.Draw(new_img)
    cv.delete("all")

def activate_event(event):
    global lastx,lasty
    cv.bind('<B1-Motion>',draw_lines)
    lastx,lasty = event.x,event.y

def draw_lines(event):
    global lastx,lasty,new_img
    x,y = event.x,event.y
    cv.create_line((lastx,lasty,x,y),width=8,fill='white',
                   capstyle=ROUND,smooth=True,splinesteps=12)
    draw.line((lastx,lasty,x,y),width=8,fill='white')
    lastx,lasty=x,y

def Recognise_Digit():
    global new_img
    img = np.array(new_img,dtype=np.uint8)
    interpolation = cv2.INTER_NEAREST
    img = cv2.resize(img,(28,28),interpolation)
    regen = model(np.reshape(img,(1,28,28,1))).astype(np.uint8)
    scale = 10
    img = img.repeat(scale,axis=1).repeat(scale,axis=0)
    regen = regen.repeat(scale,axis=1).repeat(scale,axis=0)
    # img = cv2.resize(img,(212,212),cv2.INTER_LINEAR)
    cv2.imshow("Image",img)
    cv2.imshow("rege",regen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


root = Tk()
root.resizable(1,1)
root.config(bg="black")
root.title("Handwritten Digit Recognition")

lastx,lasty = None,None
image_number = 0

encoder = tf.saved_model.load("exported_models/encoder")
decoder = tf.saved_model.load("exported_models/decoder")  

cv = Canvas(root,width = 280,height = 280,bg = 'black')
cv.grid(row=0,column=0,sticky=W,columnspan=1)
cv.bind('<Button-1>',activate_event)

new_img = Image.new("L",(280,280))
draw = ImageDraw.Draw(new_img)
btn_save = Button(text="Generate",command=Recognise_Digit,fg='black',bg='deepskyblue',width=19)
btn_save.grid(row=2,column=0,sticky=W,pady=1)
btn_clear = Button(text='Clear',command=clear_widget,fg='black',bg='deepskyblue',width=19)
btn_clear.grid(row=2,column=0,sticky=E)

root.mainloop()