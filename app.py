
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
import os 
import tensorflow as tf


from models.SwinUnet.vision_transformer import SwinUnet as ViT_seg
from models.Pix2Pix.models import Generator
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
        checkpoint = "./pretrained_ckpts/saved_model.pb"
        net = Generator()
        net = tf.saved_model.load(checkpoint)
        print("----Model Loaded-------")
        return net
       
    if model_choice == "Swin-Unet" :
        print("----Loading Swin-Unet Model-------")
        checkpoint = torch.load("pretrained_ckpts/ckpt-swin-unet.pth", map_location='cpu')
        config = _C
        config.MODEL
        net = ViT_seg(config, img_size=(448,448), num_classes=1).cuda()
        net.load_state_dict(checkpoint)
        net.eval()
        print("----Model Loaded-------")
        return net.to(device)
       
      



def predict(img, model, model_choice , thresh = 127 ):


    if model_choice == "Swin-Unet" :
  
        act = nn.Sigmoid()
        inp = torch.from_numpy(img.transpose(2,0,1)).cuda().unsqueeze(0)/255.0
        out = (act(model(inp).squeeze(0).squeeze(0)).detach().cpu().numpy()*255).astype(int)
        print(out.max(),out.mean(),out.min())
        out[out > thresh] =  255
        out[out < thresh]  = 0
        return out

    if model_choice == "Pix2pix" :
        
        inp = np.expand_dims(img,axis = 0) /255.0
        out = model(inp, training=False)
        out = (np.reshape(out[0],(256,256)) * 255).astype(int)
        print(out.max(),out.mean(),out.min())
        out[out > thresh] =  255
        out[out < thresh]  = 0
        return out
        ## inference pix2pix model
    
  

        


def main():
    
    st.title("Crack Detection")

    activities = ["Inference", "About" , "Metrics"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'Inference':
        models = ["DeepLab", "Pix2Pix" , "Swin-Unet"]
        model_choice = st.sidebar.selectbox("Select Model", models)
        st.subheader("Video Detection")

        options = ["Upload Image"]
        selection = st.selectbox("Select Option", options)

        if selection == 'Upload Image':
            image_file = st.file_uploader("Upload Image", type=['jpeg', 'png' , 'jpg'])

            if image_file is not None:

                  # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
            
                st.image(opencv_image, channels="BGR")
               
                model = load_model(model_choice)
                print(model)
                print(model_choice)
                out = predict(img=opencv_image,model=model,model_choice =model_choice , thresh=127)
             
                # fig = plot(out)

                st.text("Original Image")
                st.image(opencv_image, channels="BGR")
                st.image(out)
              
    elif choice == 'About':
        st.subheader("About Green Cover Detection App")
    
    # elif choice == "Metrics" :
    #     figs = os.listdir("./figs")
    #     names = ["loss-curve" , "confusion matrix" , "ROC_AUC curve" ,"Accuracy" , "F1" , "Precision" , "Recall"]
    #     for fig , name in zip(figs , names) :
    #         st.text(name)
    #         st.image(os.path.join("./figs",fig))


if __name__ == '__main__':
    main()
