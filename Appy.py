# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 23:01:51 2021

@author: Bryan Luna
"""

import streamlit as st
from PIL import Image
import numpy as np
import os 
# usa la  CPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow.keras as keras
from skimage.transform import resize
#caratula  



MODEL_PATH = 'D:/Tesis/DCM/DATA/Saves/save_17/Model-batch_9_witout_dec_exp.h5'

@st.cache(allow_output_mutation=True)
def cargar_modelo(MODEL_PATH):    
    model0 = keras.models.load_model(MODEL_PATH)
    return model0
model0 = cargar_modelo(MODEL_PATH)


width_shape = 512
height_shape = 512

def model_prediction(image, model):
    #img_height, img_width = 512,512
    #img = keras.preprocessing.image.load_img(image, target_size=(img_height, img_width))
    names= ['Covid 19', 'Normal', 'Pnuemonia']
    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) 
    value = names[np.argmax(score)]
    print("probablmente  implique  {} ".format(names[np.argmax(score)]))
    return value

def main(): 
    
    menu = ["Portada","Clasificación"]#,"Segmentacion","Dicomtopng"]
    choice = st.sidebar.selectbox("Menú",menu)
    #foto = 'D:/Tesis/DCM/DATA/UPload2/Covid_19/segmentacion/002822.png'
    #img_height, img_width = 512,512
    #foto = 'D:/Tesis/DCM/Prueba/Covid_19/002375.png'# 
    
    if choice == "Portada":
        
        caratula = Image.open('D:/Proyectos/Entornos/App/AHORA SI.png')
        st.image(caratula, width=700)
     
        hide = """
                <h1 
                style='text-align: center; color:#FFFFFF;'>Escuela Superior Politécnica del Chimborazo
                </h1>
                
                <h2 
                style='text-align: center; color:#FFFFFF;'>Proyecto de Investigación
                </h2>
                <h3 
                style='text-align: center; color:#FFFFFF;'>Bryan Darwin Luna Bravo
                </h3>
                
                <h4 
                style='text-align: justify;font-size:25px; color:#FFFFFF;'>RECONOCIMIENTO DE LA PRESENCIA DE SARS-COV-2 EN PULMONES A TRAVÉS DE IMÁGENES DE RADIODIAGNÓSTICO HACIENDO USO DE MACHINE LEARNIG CON LENGUAJE DE PROGRAMACIÓN PYTHON
                </h4>
                
                """
        
        st.markdown(hide, unsafe_allow_html=True)




        #st.title('Escuela Superior Politecnica del Chimborazo')
        #st.header('   texto prueba                 ')
        #st.subheader('                     Bryan Luna ')

        
        #st.write('Objetivos ')
        
    elif choice == "Clasificación":
        caratula = Image.open('D:/Proyectos/Entornos/App/clas_im.png')
        st.image(caratula, width=700)
        
        #st.title('Clasificación de imágenes Médicas ')
        st.header(' Usted puede realizar un diagnóstico previo de manera computacional gracias a un modelo de Machine Learning ')
        
        foto_2 = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])    #
        #image = keras.preprocessing.image.load_img(foto, target_size=(img_height, img_width)) #np.array(Image.open(foto))
        #img_array = keras.preprocessing.image.img_to_array(image)
        if foto_2 is not None:
            image2 = Image.open(foto_2)
            
            mostrar = np.array(image2)
            image_2 = image2.resize((512,512))
            image_2 = image_2.convert("RGB")
            #image_2 = resize(image_2,(width_shape, height_shape))
            img_array2 = keras.preprocessing.image.img_to_array(image_2)
            st.image(image2, caption="Imagen a diagnosticar", use_column_width=False)
        #  predicción
        if st.button("Reconocer"):
            impr = model_prediction(image_2,  model0)
            if impr == 'Covid 19':
                st.success(f'Después del análisis de la imágen existe la probabilidad de:  {impr}')
            if impr == 'Pnuemonia':
    
                st.success(f'Después del análisis de la imágen existe la probabilidad de:  {impr} producto de Covid 19' )
            elif impr == 'Normal':
                st.success(f'Después del análisis de la imágen existe la probabilidad  de que el pasiente este en un estado :  {impr} ' )
    
    #elif choice == "Segmentacion":
     #   st.write('Segmentacion de  imagenes  médicas ')
        
    #elif choice == "Dicomtopng":
     #   st.write('Covertir imagenes DICOM a Imagenes PNG  ')
    
if __name__ == '__main__':
    main()