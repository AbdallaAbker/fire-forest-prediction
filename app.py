import streamlit as st
import numpy as np
import yaml
import joblib
from PIL import Image

class NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

params_path = "params.yaml"

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array")
    
    if not np.issubdtype(data.dtype, np.number):
        raise NotANumber()

    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]

    if prediction == 0:
        prediction = "Safe!! Enjoy Your Trip!!"
    else:
        prediction = "Fire!! Dangerous Conditions"

    return prediction 

def validate_input(dict_request):
    try:
        for _, val in dict_request.items():
            val = float(val)
        return True
    except ValueError:
        return False

# Set up the Streamlit page
st.set_page_config(page_title="Forest Fire Prediction App", layout="wide")
# st.markdown("""
#     <style>
#     .main {
#         background-color: #000000;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# Displaying the image
image = Image.open("models/app_picture.png")
st.image(image, caption='App Image')

# Input form for the model variables
with st.form(key='ml_form'):
    st.write("Enter the input values for the prediction:")
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input('Temperature', format="%.2f")
    with col2:
        oxygen = st.number_input('Oxygen', format="%.2f")
    with col3:
        humidity = st.number_input('Humidity', format="%.2f")

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    response = ""
    dict_request = {"Temperature": temperature, "Oxygen": oxygen, "Humidity": humidity}
    if validate_input(dict_request):
        data = np.array(list(dict_request.values()), dtype=float).reshape(1, -1)
        response = predict(data)
    else:
        response = "Invalid input. Please enter valid numbers."
    st.success(f"Prediction: {response}")
