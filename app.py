import streamlit as st
from PIL import Image
import joblib 
import torch 
import torchvision.transforms as transforms 
import torch.nn.functional as F 
from model_utils import MalariaNet
from pathlib import Path
from gfile import download_model

@st.cache_resource
def load_model():     
    f_checkpoint = Path("joblib_Malarianet.joblib")
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            download_model()
    model = joblib.load('joblib_Malarianet.joblib')
    model.to(torch.device('cpu'))
    model.eval()
    return model

model = load_model() 

# Defining image transformations 
transform = transforms.Compose([
    transforms.Resize((120,120)),
    transforms.ColorJitter(0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocessing the image 
def preprocess_image(image): 
    image = transform(image).unsqueeze(0) # Add batch dimension 
    return image 

# Now lets make some predictions! 
def predict_class(image): 
    processed_image = preprocess_image(image)

    with torch.no_grad(): 
        prediction = model(processed_image) 
    # Converting the output tensor to probabilities     
    probabilities = F.softmax(prediction, dim=1) 
    # Returning the predicted class 
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class

# Streamlit UI 
def render_home_page(): 
    st.title("Malaria Detector App")
    st.markdown(
        '''
    This application enables users to detect malaria infection in individuals by analyzing images of their blood samples. Leveraging deep learning, this model achieves an accuracy rate of <span style="color:#ff8000; font-weight:bold;">96.59%</span> in identifying malaria-infected samples.
    ''', unsafe_allow_html=True

    )
    st.write("Upload an image")

    # File uploader 
    uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "png"])

    if uploaded_file is not None: 
        #Display the uploaded image 
        input_image = Image.open(uploaded_file)
        input_image = input_image.resize((200,200)) 
        col1, col2 = st.columns([1, 2])
        with col1: 
            st.image(input_image, caption = "Uploaded Image", use_column_width = True)
        # Lets make a prediction when the button is clicked 
        if st.button("Detect Malaria"):     
            with col2:  
                with st.spinner('Detecting...'): 
                    predicted_class = predict_class(input_image)           
                    #st.image(input_image, caption='Uploaded Image.', use_column_width=True)
                    if predicted_class == 1: 
                        st.success('No Malaria Detected!')
                        st.write("However should symptoms persist, please seek medical attention from a qualified healthcare professional, as they may be indicative of another illness. Take care and have a wonderful day!")
                    else: 
                        st.error('Malaria Detected.')
                        st.write("Please consult a healthcare provider for further evaluation and treatment immedidately. Malaria is a serious illness that requires medical attention. In the meantime, ensure you rest and stay hydrated.")

def render_about_page(): 
    st.title("About page")
    st.write("""
        The Malaria Detector App is a tool designed to assist users in identifying potential cases of malaria using images of blood samples. Our app employs a deep learning algorithm with a **96.59%** accuracy to analyze these images and provide quick, preliminary assessments.

        The primary goal of the Malaria Detector App is to offer a convenient and accessible means for individuals to screen for malaria symptoms, particularly in regions where access to healthcare resources may be limited. This app makes leveraging advance technology accesible and empowers uses to take proactive steps towards their health and well-being.

        ### **Disclaimer**\n
        While the Malaria Detector App provides valuable insights based on advanced algorithms, it is not a substitute for professional medical advice or diagnosis. Users are encouraged to consult with qualified healthcare professionals for accurate diagnosis and treatment recommendations.
      
            """)
    st.write("#### Example Usage")
    st.write('''
             - Upload an image of a blood sample for analysis.
             - Receive instant feedback on the likelihood of malaria infection.''')

    #Load example images
    image1 = Image.open("example_healthy.png")
    image2 = Image.open("example_parasitized.png")
    
    col3, col4 = st.columns(2)
    with col3: 
        st.image(image1, caption = "A healthy cell!", use_column_width=True)
    with col4:
        st.image(image2, caption = "A parasitized cell.", use_column_width=True)

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "About"])

if selection == "Home": 
    render_home_page()
elif selection == "About": 
    render_about_page()



