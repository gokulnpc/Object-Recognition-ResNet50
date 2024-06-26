import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
  

loaded_model = load_model('my_model.h5')
def process_image(image):
    # with keras
    img = keras.preprocessing.image.load_img(image, target_size=(32, 32))
    img = keras.preprocessing.image.img_to_array(img)
    # expected shape=(None, 224, 224, 3)
    img = tf.image.resize(img, [32, 32])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    
    
    return img

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Object Recognition using ResNet50')


    # User inputs: image
    image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    # preds = ['airplane' 'automobile' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship''truck']
    preds = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    if st.button('Predict'):
        with st.spinner('Model working....'):
            img_array = process_image(image)
            prediction = loaded_model.predict(img_array).argmax()
            st.success(f'The image is a {preds[prediction]}')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'object_recognition_model.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="object_recognition_model.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Object-Recognition-ResNet50)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This is a web app to predict the object in an image using the ResNet50 model.')
    st.write('The ResNet50 model is a pre-trained model on the ImageNet dataset.')
    st.write('The model is trained to classify 10 different objects.')
    # preds = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    st.write('The objects are: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck')
    
    
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Object-Recognition-ResNet50)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
