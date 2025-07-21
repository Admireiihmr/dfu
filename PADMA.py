import threading
import base64
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from streamlit_extras.badges import badge

# --- Custom CSS for Modern UI ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #e0e7ff 0%, #f8fafc 100%) !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        border-radius: 18px;
        background: #fff;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.12);
    }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75em 2em;
        font-size: 1.1em;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(56,189,248,0.12);
        transition: background 0.3s, transform 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #38bdf8 0%, #2563eb 100%);
        transform: translateY(-2px) scale(1.03);
    }
    .stTextInput > div > input, .stNumberInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 1.5px solid #cbd5e1;
        padding: 0.5em 1em;
        font-size: 1em;
        background: #f1f5f9;
        transition: border 0.2s;
    }
    .stTextInput > div > input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border: 1.5px solid #38bdf8;
        background: #fff;
    }
    .stSelectbox > div {
        border-radius: 8px !important;
        border: 1.5px solid #cbd5e1 !important;
        background: #f1f5f9 !important;
    }
    .stRadio > div {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #22d3ee 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1em;
        padding: 0.7em 2em;
        border: none;
        margin-top: 1em;
        box-shadow: 0 2px 8px rgba(34,211,238,0.12);
        transition: background 0.3s, transform 0.2s;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #22d3ee 100%);
        transform: translateY(-2px) scale(1.03);
    }
    .stImage > img {
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(56,189,248,0.10);
    }
    .stSubheader, .stTitle {
        color: #2563eb;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2563eb;
        font-weight: 700;
    }
    .stAlert {
        border-radius: 8px;
    }
    .st-bb {
        background: #f1f5f9 !important;
        border-radius: 12px !important;
        padding: 1em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def process_image(image):
    image = image.convert("RGB").resize((224, 224))
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
    processed_img = cv2.merge((b, g, r))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    return image, Image.fromarray(processed_img)

def predict_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json={"file": img_data})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def generate_file_content(results):
    file_content = "Prediction Results:\n"
    for label, prob in zip(results["labels"], results["probabilities"]):
        file_content += f"{label}: {float(prob):.2f}%\n"
    return file_content

def main():
    st.image("logo.png", use_container_width=True)
    st.title("Diabetic Foot Ulcer Monitoring and Severity Assessment - PADMA")
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:2em;'>
            <span style='font-size:1.2em; color:#64748b;'>Empowering diabetic foot care with AI-driven insights.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "page" not in st.session_state:
        st.session_state.page = "Registration"
    def navigate_to(page_name):
        st.session_state.page = page_name
    # --- Registration Page ---
    if st.session_state.page == "Registration":
        with st.container():
            st.subheader("User Registration")
            st.markdown("<span style='color:#e02424;'>*</span> Required fields", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Full Name *", key="full_name", help="Enter your full name.")
            with col2:
                st.text_input("Email *", key="email", help="Enter your email address.")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Next", use_container_width=True):
                missing = []
                if not st.session_state.full_name:
                    missing.append("Full Name")
                if not st.session_state.email:
                    missing.append("Email")
                if missing:
                    st.markdown(
                        f"""
                        <div style='
                            background: #fff6f6;
                            border-left: 6px solid #e02424;
                            padding: 1em 1.5em;
                            border-radius: 8px;
                            margin-bottom: 1em;
                            color: #b91c1c;
                            font-size: 1.08em;
                            display: flex;
                            align-items: center;
                        '>
                            <span style='font-size:1.5em; margin-right:0.7em;'>⚠️</span>
                            <span>
                                <b>Action Required:</b> Please complete all required fields.<br>
                                <span style='color:#991b1b;'>Missing: {', '.join(missing)}</span>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    navigate_to("Personal Details")
    # --- Personal Details Page ---
    elif st.session_state.page == "Personal Details":
        with st.container():
            st.subheader("Personal Details")
            st.markdown("<span style='color:#e02424;'>*</span> Required fields", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Name *", key="name")
                st.date_input("Date of Birth *", key="dob")
                st.number_input("Age *", min_value=1, max_value=120, step=1, key="age")
            with col2:
                st.selectbox("Gender *", ["Male", "Female", "Other"], key="gender")
                st.text_input("Address *", key="address")
                st.number_input("Pincode *", min_value=0, step=1, key="pincode")
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Back", use_container_width=True):
                    navigate_to("Registration")
            with col2:
                if st.button("Next", use_container_width=True):
                    missing = []
                    if not st.session_state.name:
                        missing.append("Name")
                    if not st.session_state.dob:
                        missing.append("Date of Birth")
                    if st.session_state.age is None:
                        missing.append("Age")
                    if not st.session_state.gender:
                        missing.append("Gender")
                    if not st.session_state.address:
                        missing.append("Address")
                    if st.session_state.pincode is None:
                        missing.append("Pincode")
                    if missing:
                        st.markdown(
                            f"""
                            <div style='
                                background: #fff6f6;
                                border-left: 6px solid #e02424;
                                padding: 1em 1.5em;
                                border-radius: 8px;
                                margin-bottom: 1em;
                                color: #b91c1c;
                                font-size: 1.08em;
                                display: flex;
                                align-items: center;
                            '>
                                <span style='font-size:1.5em; margin-right:0.7em;'>⚠️</span>
                                <span>
                                    <b>Action Required:</b> Please complete all required fields.<br>
                                    <span style='color:#991b1b;'>Missing: {',  '.join(missing)}</span>
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        navigate_to("Medical Questionnaire")
    # --- Medical Questionnaire Page ---
    elif st.session_state.page == "Medical Questionnaire":
        with st.container():
            st.subheader("Medical Questionnaire")
            st.markdown("<span style='color:#e02424;'>*</span> Required fields", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            high_bp = st.radio("Do you have high blood pressure? *", options=["Yes", "No"], key="high_bp", index=None)
            diabetes = st.radio("Do you have diabetes? *", options=["Yes", "No"], key="diabetes", index=None)
            diabetes_years = None
            if diabetes == "Yes":
                diabetes_years = st.number_input("How many years have you had diabetes? *", min_value=0, step=1, key="diabetes_years")
            else:
                st.radio("Do you have ulcers on your foot?", options=["Yes", "No"], key="ulcers")
            st.text_area("Please list any other current medical conditions:", key="medical_conditions")
            st.text_area("Are you taking any medications?", key="medications")
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Back", use_container_width=True):
                    navigate_to("Personal Details")
            with col2:
                if st.button("Next", use_container_width=True):
                    missing = []
                    if high_bp is None:
                        missing.append("High Blood Pressure")
                    if diabetes is None:
                        missing.append("Diabetes")
                    if diabetes == "Yes" and (diabetes_years is None):
                        missing.append("Years with Diabetes")
                    if missing:
                        st.markdown(
                            f"""
                            <div style='
                                background: #fff6f6;
                                border-left: 6px solid #e02424;
                                padding: 1em 1.5em;
                                border-radius: 8px;
                                margin-bottom: 1em;
                                color: #b91c1c;
                                font-size: 1.08em;
                                display: flex;
                                align-items: center;
                            '>
                                <span style='font-size:1.5em; margin-right:0.7em;'>⚠️</span>
                                <span>
                                    <b>Action Required:</b> Please complete all required fields.<br>
                                    <span style='color:#991b1b;'>Missing: {', '.join(missing)}</span>
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        navigate_to("Image Upload")
    # --- Image Upload and Prediction Page ---
    elif st.session_state.page == "Image Upload":
        with st.container():
            st.subheader("Image Upload and Prediction")
            st.markdown("<hr>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG) *", type=["jpg", "jpeg", "png"])
            if st.button("Back", use_container_width=True):
                navigate_to("Medical Questionnaire")
            if uploaded_file:
                image = Image.open(uploaded_file)
                _, processed_image = process_image(image)
                st.image(processed_image, caption="Processed Image for Prediction", use_container_width=True)
                if st.button("Predict", use_container_width=True):
                    predictions = predict_image(processed_image)
                    if predictions and "probabilities" in predictions:
                        st.success("Prediction Complete!", icon="✅")
                        st.subheader("Predictions:")
                        for label, prob in zip(predictions["labels"], predictions["probabilities"]):
                            st.markdown(f"<div style='padding:0.5em 1em; background:#f1f5f9; border-radius:8px; margin-bottom:0.5em; font-size:1.1em;'><b>{label}</b>: {float(prob):.2f}%</div>", unsafe_allow_html=True)
                        file_content = generate_file_content(predictions)
                        st.download_button(
                            label="Save Results",
                            data=file_content,
                            file_name="prediction_results.txt",
                            mime="text/plain",
                        )
            elif st.button("Predict", use_container_width=True):
                st.markdown(
                    f"""
                    <div style='
                        background: #fff6f6;
                        border-left: 6px solid #e02424;
                        padding: 1em 1.5em;
                        border-radius: 8px;
                        margin-bottom: 1em;
                        color: #b91c1c;
                        font-size: 1.08em;
                        display: flex;
                        align-items: center;
                    '>
                        <span style='font-size:1.5em; margin-right:0.7em;'>⚠️</span>
                        <span>
                            <b>Action Required:</b> Please upload an image before predicting.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
def start_fastapi():
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=5000)

if __name__ == "__main__":
    threading.Thread(target=start_fastapi, daemon=True).start()
    main()
