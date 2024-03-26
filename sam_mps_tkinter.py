import cv2
import tkinter as tk
from PIL import Image, ImageTk
## ----------
import numpy as np
import torch
import matplotlib.pyplot as plt

MASK_COLOR = np.array([255, 0, 0])

# Load the Lena image ---------------------------
image = cv2.imread("/Users/jup1/Downloads/lena512gray.jpg")

import sys
sys.path.append("..")

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/Users/jup1/Downloads/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "mps"  # 변경된 부분: "cpu" -> "mps"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

## 마스크를 이미지로 변환하는 함수를 정의합니다.
def make_mask_2_img(mask):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * MASK_COLOR.reshape(1, 1, -1)
    mask_image = mask_image.astype(np.uint8)
    return mask_image

def on_mouse_move(event):
    global image
    input_point = np.array([[event.x, event.y]])
    input_label = np.array([1])
    mask, _,_ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    binary_img = make_mask_2_img(mask)
    # Convert color space from BGR to RGB
    b_image = cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB)
    # Convert the image format for tkinter
    im = Image.fromarray(b_image)
    img = ImageTk.PhotoImage(im)
    # Update the tkinter label with the binary image
    img_label_proc.img = img
    img_label_proc.config(image=img)

## ===============================
root = tk.Tk()

# Convert color space from BGR to RGB for initial image display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
im = Image.fromarray(image_rgb)
img = ImageTk.PhotoImage(im)

# Create a tkinter label to display the original image
img_label_orig = tk.Label(root, image=img)
img_label_orig.grid(row=0, column=0)  # place it in position 00

# Create a second label for processed image display
img_label_proc = tk.Label(root)
img_label_proc.grid(row=0, column=1)  # place it in position 01

root.bind("<Button-1>", on_mouse_move)
root.mainloop()