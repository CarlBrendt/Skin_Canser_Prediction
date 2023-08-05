import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import streamlit as st
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image


voc_dict = {
    'Actinic keratoses' : 'Actinic keratosis is a rough, scaly patch or bump on the skin. It’s also known as a solar keratosis. Actinic keratoses are very common, and many people have them. They are caused by ultraviolet (UV) damage to the skin. Some actinic keratoses can turn into squamous cell skin cancer. Because of this, the lesions are often called precancer. They are not life-threatening. But if they are found and treated early, they do not have the chance to develop into skin cancer.', 
    'Basal cell carcinoma' : 'Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off.Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms. Basal cell carcinoma occurs most often on areas of the skin that are exposed to the sun, such as your head and neck.Most basal cell carcinomas are thought to be caused by long-term exposure to ultraviolet (UV) radiation from sunlight. Avoiding the sun and using sunscreen may help protect against basal cell carcinoma.', 
    'Benign keratosis-like lesions' : 'A seborrheic keratosis (seb-o-REE-ik ker-uh-TOE-sis) is a common noncancerous (benign) skin growth. People tend to get more of them as they get older.Seborrheic keratoses are usually brown, black or light tan. The growths (lesions) look waxy or scaly and slightly raised. They appear gradually, usually on the face, neck, chest or back.',
    'Dermatofibroma' : "Dermatofibroma is a commonly occurring cutaneous entity usually centered within the skin's dermis. Dermatofibromas are referred to as benign fibrous histiocytomas of the skin, superficial/cutaneous benign fibrous histiocytomas, or common fibrous histiocytoma. These mesenchymal cell lesions of the dermis clinically are firm subcutaneous nodules that occur on the extremities in the vast majority of cases and may or may not be associated with overlying skin changes. They are most commonly asymptomatic and usually relatively small, less than or equal to 1 centimeter in diameter.",  
    'Melanoma' : 'Melanoma, also redundantly known as malignant melanoma is a type of cancer that develops from the pigment-producing cells known as melanocytes. Melanomas typically occur in the skin, but may rarely occur in the mouth, intestines, or eye (uveal melanoma).In women, they most commonly occur on the legs, while in men, they most commonly occur on the back. About 25% of melanomas develop from moles. Changes in a mole that can indicate melanoma include an increase in size, irregular edges, change in color, itchiness, or skin breakdown.', 
    'Melanocytic nevi' : "Melanocytic nevus is the medical term for a mole. Nevi can appear anywhere on the body. They are benign (non-cancerous) and typically do not require treatment. A very small percentage of melanocytic nevi may develop a melanoma﻿ within them. Of note, the majority of cutaneous melanomas arise within normally appearing skin.Benign nevi are usually a single color, ranging from skin-colored to dark brown. They are typically round or oval-shaped. In addition, benign moles are symmetric, that is, when a line is drawn within them, the two halves have the same appearance. Most melanocytic nevi are the size of a pencil eraser or smaller.",
    'Vascular lesions' : "Vascular lesions are relatively common abnormalities of the skin and underlying tissues, more commonly known as birthmarks. There are three major categories of vascular lesions: Hemangiomas, Vascular Malformations, and Pyogenic Granulomas."
}
@st.cache_data()
def load_model():
    model_path = 'skin_model.h5'
    pretrained_model = tf.keras.models.load_model(model_path)
    return pretrained_model


with st.spinner('Model is being loaded..'):
   pretrained_model=load_model()


def load_image():
    
    uploaded_file = st.file_uploader(
        label='Upload Image')
    
    if uploaded_file is not None:
        
        image_data = uploaded_file.getvalue()

        st.image(image_data)

        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
def preprocess_image(img):
    
    SIZE = 64
    img = img.resize((SIZE, SIZE))
    img  = image.img_to_array(img)
    img = img / 255
    return np.expand_dims(img, axis=0)

classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']

def getPrediction(img):
    
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
        
    
    pred = pretrained_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    
    print("Diagnosis is:", pred_class)
    
    return pred_class

def main():
    
    st.title('Skin Cancer Lesion Classifier')
    review = '<p style="font-family:sans-serif; color:Green; font-size: 20px;">Upload an image and click submit</p>'
    st.markdown(review, unsafe_allow_html=True)
    
    img = load_image()

    result = st.button('Submit')

    if result:

        x = preprocess_image(img)

        preds = getPrediction(x)

        st.write('**Results of prediction:**')

        st.write(f'It is {preds}')
        st.write(voc_dict[preds])
    
if __name__ == '__main__':
    main()