import streamlit as st
import torch
import torch.nn.functional as F
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import csv
import os
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Page Configuration
st.set_page_config(page_title="MindBridge AI", layout="centered")

# CSS STYLING
st.markdown(
    """
    <style>


    <style>
    [data-testid="stSidebarCollapseButton"], 
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] * {
        color: #005A9C !important;
    }
    
    /* 1. Global Page Style */
    .stApp {
        background-color: #f4f7f6 !important; 
    }
    .block-container {
        padding-top: 2rem !important; 
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important; 
    }
   
  
    /* 2. Typography */
    h1, h2, h3 {
        color: #005A9C !important; 
    }
    label {
        color: #333333 !important; 
        font-weight: 600 !important;
    }
    p, .stMarkdown, .stCaption {
        color: #444444 !important; 
    }

    /* 3. Sidebar Style */
    [data-testid="stSidebar"] {
        background-color: #005A9C !important;
    }
    [data-testid="stSidebar"] * {
        color: White !important; 
    }

    /* 4. Buttons */
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
        background-color: #005A9C !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    div[data-testid="stButton"] button {
        width: 100% !important;
    }
    div[data-testid="stButton"] button p, div[data-testid="stFormSubmitButton"] button p {
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        margin: 0 !important;
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #004080 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* 5. Inputs (Text Area, Selectbox, Input) */
    .stTextArea textarea, .stTextInput input, div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #333333 !important;
        border: 1px solid #ced4da !important;
        border-radius: 8px !important;
    }
    ul[role="listbox"], li[role="option"] {
        background-color: white !important;
        color: #333333 !important;
    }

    /* 6. Result Card Style */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 32px !important;
        font-weight: bold;
        color: white !important;
        margin: 20px 0;
        line-height: 1.2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .english-label {
        font-size: 20px;
        font-weight: 500;
        opacity: 0.9;
        display: block;
        margin-top: 5px;
        margin-bottom: 10px;
        color: white !important;
    }
    .severe { background-color: #FF4B4B; }
    .moderate { background-color: #FF9B4B; }
    .mild { background-color: #4B9BFF; }
    .normal { background-color: #00CC96; }
    
    /* 7. Form container background */
    [data-testid="stForm"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 8. XAI Expander Highlight */
    [data-testid="stExpander"] {
        border: 2px solid #005A9C !important; 
        border-radius: 10px !important;
        background-color: white !important;
        box-shadow: 0 4px 8px rgba(0, 90, 156, 0.15) !important; 
        margin-top: 15px;
        margin-bottom: 25px;
    }
    [data-testid="stExpander"] summary {
        background-color: #f0f7ff !important; 
        border-radius: 8px 8px 0 0 !important;
        padding: 15px !important;
    }
    [data-testid="stExpander"] summary p {
        color: #005A9C !important; 
        font-weight: bold !important;
        font-size: 18px !important;
    }
    [data-testid="stExpander"] summary svg {
        fill: #005A9C !important; 
    }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("<h1>MindBridge AI</h1>", unsafe_allow_html=True)
    st.markdown("### About the Project")
    st.write("An Explainable AI tool for Mental Health Insights in Sinhala & Singlish Code-Mixed Text.")
    st.markdown("---")
    st.markdown("### How to use:")
    st.write("1. අදාළ සමාජ මාධ්‍ය සටහන (Social Media Post) ඇතුළත් කරන්න.")
    st.write("2. **'Analyze & Explain'** බොත්තම ඔබන්න.")
    st.write("3. ප්‍රතිඵලය සහ AI එකට බලපෑ වචන අධ්‍යයනය කරන්න.")
    st.markdown("---")
    st.caption("Developed by: Eshan Hasitha")

# MAIN APP
st.markdown("<h1>MindBridge AI</h1>", unsafe_allow_html=True)
st.markdown("### Mental Health Text Analyzer")

@st.cache_resource
def load_model():
    model_path = "EshanHasitha/mindbridge-xlm-robertamodel"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

def predictor(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

with st.container():
    st.markdown("<label>සමාජ මාධ්‍ය සටහන මෙහි ඇතුළත් කරන්න (Enter Social Media Post here):</label>", unsafe_allow_html=True)
    
    #Added label="Post Input" and label_visibility="collapsed"
    user_input = st.text_area("Post Input", height=150, label_visibility="collapsed")
    
    prediction = 0
    confidence = 0.0
    labels_english = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    
    if st.button("Analyze & Explain", type="primary"):
        if user_input.strip() == "":
            st.warning("කරුණාකර සමාජ මාධ්‍ය සටහනක් ඇතුළත් කරන්න!")
        else:
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item() * 100
                st.session_state['last_prediction'] = prediction 

            severity_class = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}
            labels_sinhala = {0: "සාමාන්‍ය තත්ත්වය", 1: "සුළු පීඩනයක්", 2: "මධ්‍යම පීඩනයක්", 3: "දැඩි පීඩනයක්"}
            
            st.markdown(f"""
                <div class="result-box {severity_class[prediction]}">
                    {labels_sinhala[prediction]}<br>
                    <span class="english-label">({labels_english[prediction]})</span>
                    <span style="font-size: 16px; font-weight: normal;">Confidence: {confidence:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)

            if prediction == 3:
                st.error("**CRITICAL ALERT:** දැඩි මානසික අවපීඩනයක් හඳුනාගන්නා ලදී. කරුණාකර වහාම 1333 අමතන්න.")

            with st.expander("AI තීරණයට හේතු වූ කරුණු (Detailed Explanation)"):
                with st.spinner("Analyzing text with LIME..."):
                    explainer = LimeTextExplainer(class_names=["Normal", "Mild", "Moderate", "Severe"])
                    exp = explainer.explain_instance(user_input, predictor, num_features=6, top_labels=1, num_samples=500)
                    
                    lime_html = f"""
                    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px;">
                        {exp.as_html()}
                    </div>
                    """
                    components.html(lime_html, height=450, scrolling=True)

    #HUMAN-IN-THE-LOOP FEEDBACK
    st.markdown("---")
    st.markdown("### Expert Feedback")
    st.caption("AI ප්‍රතිඵලය නිවැරදි නොවේ නම්, කරුණාකර නිවැරදි තත්ත්වය පහතින් යාවත්කාලීන කරන්න.")
    
    is_professional = st.checkbox("ඔබ මානසික සෞඛ්‍ය වෘත්තිකයෙක් නම් හරි සලකුණ ලකුණු කරන්න (Tick here if you are a Mental Health Professional)")
    
    with st.form("feedback_form"):
        doc_name = ""
        doc_profession = ""
        doc_contact = ""
        
        if is_professional:
            st.markdown("**වෘත්තීය තොරතුරු (Professional Details):**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<label>ඔබගේ නම (Name)</label>", unsafe_allow_html=True)
                #Added label and visibility
                doc_name = st.text_input("Name", key="doc_name", label_visibility="collapsed")
            with col2:
                st.markdown("<label>තනතුර (Designation / Profession)</label>", unsafe_allow_html=True)
                #Added label and visibility
                doc_profession = st.text_input("Profession", key="doc_profession", label_visibility="collapsed")
            
            st.markdown("<label>දුරකථන අංකය (Contact Number)</label>", unsafe_allow_html=True)
            #Added label and visibility
            doc_contact = st.text_input("Contact", key="doc_contact", label_visibility="collapsed")
            st.markdown("---")
        
        st.markdown("<label>නිවැරදි තත්ත්වය කුමක්ද? (True Label)</label>", unsafe_allow_html=True)
        #Added label and visibility
        correct_label = st.selectbox(
            "True Label", 
            ["AI එක නිවැරදියි (Correct)", "සාමාන්‍ය තත්ත්වය (Normal)", "සුළු පීඩනයක් (Mild)", "මධ්‍යම පීඩනයක් (Moderate)", "දැඩි පීඩනයක් (Severe)"],
            label_visibility="collapsed"
        )
        
        st.markdown("<label>අතිරේක සටහන් (Optional Comments):</label>", unsafe_allow_html=True)
        #Added label and visibility
        comments = st.text_area("Comments", height=70, label_visibility="collapsed")
        
        submit_btn = st.form_submit_button("Submit Feedback")
        
        if submit_btn:
            if is_professional and (not doc_name.strip() or not doc_profession.strip() or not doc_contact.strip()):
                st.error("කරුණාකර ඔබගේ නම, තනතුර සහ දුරකථන අංකය සම්පූර්ණ කරන්න.")
            elif user_input.strip() == "":
                st.error("කරුණාකර පළමුව ඉහළින් සමාජ මාධ්‍ය සටහනක් ඇතුළත් කර Analyze කරන්න.")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ai_predicted_label = labels_english.get(st.session_state.get('last_prediction', 0), "Unknown")
                
                try:
                    creds_dict = dict(st.secrets["gcp_service_account"])
                    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                    client = gspread.authorize(creds)
                
                    sheet = client.open("MindBridge_Feedback").sheet1
                
                    row_to_add = [
                        timestamp, user_input, ai_predicted_label, correct_label, 
                        "Yes" if is_professional else "No", 
                        doc_name if is_professional else "N/A", 
                        doc_profession if is_professional else "N/A", 
                        doc_contact if is_professional else "N/A", 
                        comments
                    ]
                    sheet.append_row(row_to_add)
                    st.success("Feedback submitted successfully to the system!")
                except Exception as e:
                    st.error(f"Error submitting feedback: {e}")

#DISCLAIMER
st.markdown("---")
st.caption("**Disclaimer:** MindBridge AI is an academic research tool and NOT a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a mental health crisis, please contact emergency services or call **1333** immediately.")
