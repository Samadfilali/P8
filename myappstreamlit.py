import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
import os

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

path_val_seg = './input/gtFine/val/frankfurt/'
path_val_real= './input/leftImg8bit/val/frankfurt/'


@st.cache
def load_model() :
    mdl = tf.keras.models.load_model('./model_v3_unet')
    return mdl

def file_selector(folder_path='./input/leftImg8bit/val/frankfurt/'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)



## chargement des données en supprimant la partie basse de l'image 
def LoadImage(path_file,width=IMG_WIDTH, height=IMG_HEIGHT,channels=IMG_CHANNELS, cut_bottom=200):
    img = imread(path_file)  
    img = img[:-cut_bottom, :]
    img = resize(img, (width, height))
    return img

# chargement des masks en gardant que les couches qui nous interessent
def getSegmentationArr(image_path, classes=8, width=IMG_WIDTH, height=IMG_HEIGHT,cut_bottom=200):
    img = imread(image_path) 
    img = img[:-cut_bottom, :]
    img_mask=cv2.resize(img, (width, height))
    img_mask_result=np.zeros(shape=(width, height, classes), dtype=np.uint8)
    img_mask_result[:,:,0][img_mask==2]=1  # ---car hood
    img_mask_result[:,:,1][img_mask==7]=1  # road
    img_mask_result[:,:,2][img_mask==8]=1  # sidewalk
    img_mask_result[:,:,3][img_mask==9]=1  # nature
    img_mask_result[:,:,4][img_mask==11]=1 # building
    img_mask_result[:,:,5][img_mask==12]=1 # signs
    img_mask_result[:,:,6][img_mask==24]=1 # human
    img_mask_result[:,:,7][img_mask==26]=1 # vehicle   
    return img_mask_result


# Coloration des couches des masks
def LayersToRGBImage(img):
    colors = [(125,125,125), (0,255,0), (0,0,255),
             (255,255,0), (255,0,255), (0,255,255),
             (255,255,255), (200,50,0), (0,0,0)]
    
    nimg = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[2]):
        c = img[:,:,i]
        col = colors[i]
        
        for j in range(3):
            nimg[:,:,j]+=col[j]*c
            
    nimg = nimg/255
    return nimg


st.write("""
# Semantic Segmentation Image App

""")


image = imread('seg_image_logo.jpeg')
st.image(image, use_column_width=True)

st.write("""
En entrée, cette App reçoit une image et affiche les masques réel et prédit 
""")


# st.header('Identifying digits from Images')
st.subheader('Please upload an image to segment')
# liste déroulante pour choisir une image
filename = file_selector()
image = imread(filename)
st.image(image, use_column_width=True)

if st.button('Predict'):
    model = load_model()
    # Affichage du masque réel
    st.write("""
             Le masque réel
             """)
    path_file_seg=filename.replace("leftImg8bit.png","gtFine_labelIds.png")
    path_file_seg=path_file_seg.replace("/leftImg8bit/","/gtFine/")
    image=getSegmentationArr(path_file_seg)
    image=LayersToRGBImage(image)
    st.image(image, use_column_width=True)
    
    # affichage de  la segmentation prédite
    st.write("""
             Le masque prédit
             """)
    test_im= LoadImage(filename,width=IMG_WIDTH, height=IMG_HEIGHT,channels=IMG_CHANNELS)
    x = np.expand_dims(test_im, axis=0)
    y_pred=model.predict(x, verbose=1)
    y_pred_unit8 = (y_pred > 0.2).astype(np.uint8)
    image=LayersToRGBImage(y_pred_unit8[0])
    image = image/np.amax(image)
    st.image(image, use_column_width=True)    
  

















