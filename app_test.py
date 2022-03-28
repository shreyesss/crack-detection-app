# Core Pkgs
# import ee
from email.policy import strict
from tabnanny import check
from torch.serialization import load
import streamlit as st
import cv2
import numpy as np
# import folium
import os
from PIL import Image
# from selenium import webdriver
import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px



import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
import requests


from models.SwinUnet.vision_transformer import SwinUnet as ViT_seg
# from models.Pix2Pix.models import Generator
# from models.Deeplab.deeplab import DeepLab
from configs.config import _C
from torch import nn
os.environ['KMP_DUPLICATE_LIB_OK']='True'


st.set_page_config(layout="wide")
if torch.cuda.is_available() :
    device = "cuda"
else :
    device = "cpu"


   

def load_model(model_choice):


    if model_choice == "Pix2Pix" :
        print("----Loading Pix2Pix Model-------")
        loaded = tf.saved_model.load("pretrained_ckpts/unetgenerator")
        infer = loaded.signatures["serving_default"]
        # print(list(infer.signatures.keys()))  # ["serving_default"]
        print("----Model Loaded-------")
        return infer
       
        
       
    
    if model_choice == "Swin-Unet" :
        print("----Loading Swin-Unet Model-------")
        checkpoint = torch.load("pretrained_ckpts/ckpt-swin-unet.pth", map_location='cpu')
        config = _C
        # config.MODEL
        net = ViT_seg(config, img_size=(448,448), num_classes=1).cuda()
        net.load_state_dict(checkpoint)
        net.eval()
        print("----Model Loaded-------")
        return net.to(device)
       
      



def predict(img, model, model_choice , thresh = 127 ):


    if model_choice == "Swin-Unet" :
  
        act = nn.Sigmoid()
        img = cv2.resize(img , dsize= (448,448))
        inp = torch.from_numpy(img.transpose(2,0,1)).cuda().unsqueeze(0)/255.0
        out = (act(model(inp).squeeze(0).squeeze(0)).detach().cpu().numpy()*255).astype(int)
        print(out.max(),out.mean(),out.min())
        out[out > thresh] =  255
        out[out < thresh]  = 0
        out = cv2.merge((out,out,out))
        return out

    if model_choice == "Pix2Pix" :
        
       img = cv2.resize(img , dsize= (256,256))
       inp  = (img/127.5) - 1
       inp = np.expand_dims(inp,axis=0)  
       
       pred = model(tf.constant(inp,dtype=tf.float32))["conv2d_transpose_7"]
       pred = pred.numpy()[0].reshape(256,256)
       pred = (255 * (pred * 0.5 + 0.5)).astype(int)
       pred[pred > thresh] =  255
       pred[pred < thresh]  = 0
       pred = cv2.merge((pred,pred,pred))
       return pred
        ## inference pix2pix model
    
  

        
# url = "http://10.42.166.49:8080/shot.jpg"

def main():
    
        st.title("Crack Detection")
        activities = ["Live_video" , "Photo_Upload" , "Demo"]
        choice = st.sidebar.selectbox("Select Activty", activities)

        if choice == 'Live_video':
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            models = ["Pix2Pix" , "Swin-Unet"]
            model_choice = st.sidebar.selectbox("Select Model", models)
            st.subheader("Video Detection")
            camera = cv2.VideoCapture(0)
            model = load_model(model_choice)

            while run:
                # img_resp = requests.get(url)
                # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                # frame = cv2.imdecode(img_arr, -1)
                _, frame = camera.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if model_choice == "Pix2Pix" :
                    frame = cv2.resize(frame , dsize= (256,256))
                if model_choice == "Swin-Unet" :
                    frame = cv2.resize(frame , dsize= (448,448))
                out = predict(img=frame,model=model,model_choice =model_choice , thresh=127)
                print(out.shape)
                # frame = cv2.resize(frame , dsize= (400,400))
                # out = cv2.resize(out , dsize= (400,400))
                
                disp = np.concatenate([frame , out] , axis = 1)
                FRAME_WINDOW.image(disp)
        
        
        if choice == "Photo_Upload" :
            models = [ "Pix2Pix" , "Swin-Unet"]
            model_choice = st.sidebar.selectbox("Select Model", models)
            st.subheader("Photo Detection")

            options = ["Upload Image"]
            selection = st.selectbox("Select Option", options)

            if selection == 'Upload Image':
                image_file = st.file_uploader("Upload Image", type=['jpeg', 'png' , 'jpg'])

                if image_file is not None:

                    # Convert the file to an opencv image.
                    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                
                
                    model = load_model(model_choice)
                    out = predict(img=opencv_image,model=model,model_choice =model_choice , thresh=127)             
                    if model_choice == "Pix2Pix" :
                        opencv_image = cv2.resize(opencv_image , dsize= (256,256))
                    if model_choice == "Swin-Unet" :
                        opencv_image = cv2.resize(opencv_image , dsize= (448,448))
            
                
                    # fig = plot(out)
                    result = np.concatenate([opencv_image , out] , axis = 1)
                    st.image(result, channels="BGR")


        if choice == "Demo" :
            video_file = open('Demo_output.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
                   
        
        
        else:
                st.write('Stopped')

                
            


if __name__ == '__main__':
    main()




