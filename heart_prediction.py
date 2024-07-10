import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
# Load the trained model
loaded_model = joblib.load('pipeline.joblib')



def preprocess_input(age, gender, ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, alco, active):
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'bmi': [bmi],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })

    # Apply the same preprocessing steps as during training
    input_data['gender'] = input_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    input_data['smoke'] = input_data['smoke'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['alco'] = input_data['alco'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['active'] = input_data['active'].apply(lambda x: 1 if x == 'Active' else 0)
    input_data['cholesterol'] = input_data['cholesterol'].map({'Low': 1, 'Medium': 2, 'High': 3})
    input_data['gluc'] = input_data['gluc'].map({'Low': 1, 'Medium': 2, 'High': 3})

    return input_data

# Streamlit UI
st.title('Heart Disease Prediction App')
watson_assistant_script = """
  <script>
  window.watsonAssistantChatOptions = {
    integrationID: "435cfc1d-9624-474b-865f-3d71afc9b027", // The ID of this integration.
    region: "au-syd", // The region your integration is hosted in.
    serviceInstanceID: "5fc6f31e-5083-4fd2-a6e1-3aeb31784224", // The ID of your service instance.
    onLoad: async (instance) => { await instance.render(); }
  };
  setTimeout(function(){
    const t=document.createElement('script');
    t.src="https://web-chat.global.assistant.watson.appdomain.cloud/versions/" + (window.watsonAssistantChatOptions.clientVersion || 'latest') + "/WatsonAssistantChatEntry.js";
    document.head.appendChild(t);
  });
</script>
"""
# Collect user input with Streamlit widgets on the main page
st.sidebar.header('User Input Features')
age = st.sidebar.slider('Age', 18, 100, 25)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
ap_hi = st.sidebar.slider('Systolic Blood Pressure (mm Hg)', 80, 250, 120)
ap_lo = st.sidebar.slider('Diastolic Blood Pressure (mm Hg)', 50, 150, 80)
bmi = st.sidebar.slider('Body Mass Index (BMI)', 15.0, 45.0, 25.0)
cholesterol = st.sidebar.selectbox('Cholesterol Level', ["Low", "Medium", "High"])
gluc = st.sidebar.selectbox('Glucose Level', ["Low", "Medium", "High"])
smoke = st.sidebar.selectbox('Smoking', ['Yes', 'No'])
alco = st.sidebar.selectbox('Alcohol Consumption', ['Yes', 'No'])
active = st.sidebar.selectbox('Physical Activity', ['Active', 'Inactive'])

# Preprocess user input
input_data = preprocess_input(age, gender, ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, alco, active)

# Make predictions
if st.button('Predict'):
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:
        st.write('The model predicts that the individual has a high chance of heart problems.')
    else:
        st.write('The model predicts that the individual has a low chance of heart problems.')

components.html(watson_assistant_script, height=450)
