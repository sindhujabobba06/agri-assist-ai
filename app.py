import streamlit as st
from google.genai import Client, types
from PIL import Image
from gtts import gTTS
import io

st.set_page_config(page_title="AgriAssist AI", page_icon="🌾")
st.title("🌾 AgriAssist: Multilingual AI Farmer Advisor")

# REPLACE WITH YOUR ACTUAL GEMINI API KEY
API_KEY = st.secrets["GEMINI_API_KEY"]

client = Client(api_key=API_KEY)

# Language Selection Mapping
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Marathi": "mr",
    "Kannada": "kn"
}

st.sidebar.header("Language Settings")
selected_language_name = st.sidebar.selectbox(
    "Select Language / भाषा चुनें", 
    list(language_codes.keys())
)
lang_code = language_codes[selected_language_name]

st.write(f"Ask any farming question via **text, voice, or leaf photo** in **{selected_language_name}**.")

# 1. Leaf Image Input
uploaded_image = st.file_uploader("Upload Leaf/Crop Photo for Disease Diagnosis", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Crop Image", use_container_width=True)

# 2. Voice Input Widget
audio_recording = st.audio_input("🎤 Speak your question (Click mic to record)")

# 3. Text Input
user_prompt = st.text_input("OR type your farming issue here:")

# Submit Button
if st.button("Get AI Advice"):
    if not user_prompt and not uploaded_image and not audio_recording:
        st.warning("Please provide a text question, a voice recording, or an image.")
    else:
        with st.spinner("Analyzing agricultural inputs..."):
            system_instruction = f"""
            You are an expert agricultural assistant helping local farmers.
            Provide accurate, simple, and practical advice on crop health, pests, and soil management.
            Respond strictly in the following language: {selected_language_name}.
            Keep response clean without complex symbols so it reads well out loud.
            """
            
            prompt_contents = []

            if uploaded_image:
                prompt_contents.append(image)

            if audio_recording:
                audio_bytes = audio_recording.read()
                prompt_contents.append(
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=audio_recording.type
                    )
                )

            if user_prompt:
                prompt_contents.append(user_prompt)
            elif audio_recording and not uploaded_image:
                prompt_contents.append("Listen to this voice recording from a farmer and answer their agricultural query.")
            elif uploaded_image and not (user_prompt or audio_recording):
                prompt_contents.append("Identify the disease or issue in this crop photo and suggest immediate treatments.")

            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt_contents,
                    config={"system_instruction": system_instruction}
                )
                
                advice_text = response.text
                st.markdown("### 📋 Recommended Action Plan:")
                st.write(advice_text)

                # Generate Audio Output (TTS)
                try:
                    tts = gTTS(text=advice_text, lang=lang_code, slow=False)
                    sound_file = io.BytesIO()
                    tts.write_to_fp(sound_file)
                    st.markdown("### 🔊 Listen to Advice:")
                    st.audio(sound_file, format="audio/mp3")
                except Exception as tts_error:
                    st.warning(f"Text-to-Speech unavailable for this language formatting: {tts_error}")

            except Exception as e:
                st.error(f"Error connecting to AI service: {e}")