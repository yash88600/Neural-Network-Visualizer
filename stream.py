
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt

@st.cache(persist = True)
def model_load(path):
    layers =[]
    model = load_model(path)
    for l in model.layers:
        layers.append(l.name)
    outputs = [layer.output for layer in model.layers[1:]]
    model = Model(model.input,outputs)
    ch = [layer.name for layer in model.layers if len(layer.output_shape)==4]

    return layers,model,ch
@st.cache(persist = True)
def image_processing(image,img_shape):
    image = load_img(image, target_size=(img_shape[1],img_shape[2]))
    image = np.array(image)
    if img_shape[3]==1 and image.shape[2]==3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if img_shape[3]==3 and image.shape[2]==1:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    image = np.expand_dims(image,axis=0)
    return image
        



st.title('Neural Network Visualizer')
st.markdown('Neural Network Visualizer to visualize the output of each layer')
st.sidebar.title('Neural Network Visualizer')
st.sidebar.markdown('Neural Network Visualizer to visualize the output of each layer')

path = st.text_input('Enter the path to the model')
if len(path)>0:
    layers,model,av_layer = model_load(path)
    
    layer_visual = st.sidebar.selectbox('Choose a layer to visualize',av_layer)
    image = st.sidebar.file_uploader("Upload the image",type='jpg')
    layer_number = layers.index(layer_visual) 
    
    if image is not None:
        st.sidebar.image(image,width = 300)
        image = image_processing(image,model.input.shape)
        prediction = model.predict(image)
        output = prediction[-1]
        
        st.sidebar.markdown('Image belongs to class : '+str(np.argmax(output)))
        mat = prediction[layer_number]
        mat = mat.reshape(mat.shape[1],mat.shape[2],mat.shape[3])
        
        st.sidebar.subheader("filter Visualization")
        filters = st.sidebar.radio('choose...',('Single filter','All filters at once','Subset of filters'))
        
        if filters == 'Single filter':
            c= st.sidebar.number_input("Enter the filter Number",0.0,float(mat.shape[2]-1),step=1.0,key='c')
            title = str(layer_visual) +' shape : ' +str(mat[:,:,int(c)].shape)
            plt.title(title)
            plt.imshow(mat[:, :,int(c)])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()
            st.pyplot()
            
        if filters == 'All filters at once':
            st.markdown(layer_visual +' layer has '+ str(mat.shape[2]) + ' filters')
            row = int(mat.shape[2]/5)+1
            plt.figure(num=None, figsize=(10, row*3),dpi=100,facecolor='w', edgecolor='k')
            for i in range(0,mat.shape[2]):
                plt.subplot(row,5,i+1)
                plt.title(i+1)
                plt.imshow(mat[:, :,i])
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            st.pyplot()
            
        if filters == 'Subset of filters':
            num1= st.sidebar.number_input("From filter no",0.0,float(mat.shape[2]-2),step=1.0,key='num1')
            num2= st.sidebar.number_input("To filter no",1.0,float(mat.shape[2]-1),step=1.0,key='num2')
            row = int((num2 - num1)/5)+1
            plt.figure(num=None, figsize=(10, row*3),dpi=100,facecolor='w', edgecolor='k')
            for i in range(int(num1)-1,int(num2)):
                plt.subplot(row,5,i-int(num1)+2)
                plt.title(i+1)
                plt.imshow(mat[:, :,i])
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            st.pyplot()
