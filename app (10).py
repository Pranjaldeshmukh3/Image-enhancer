import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('image_enhancer_model.h5')
    return model

model = load_model()

# Title
st.title("🧠 Image Enhancer - Low Light to Clear Vision")

st.markdown("""
Upload a **low-light image**, and this app will enhance it using a trained deep learning model.
""")

# Upload image
uploaded_file = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction."""
    image = image.convert('RGB')  # Convert to RGB if not already
    image = image.resize((500, 500))  # Resize image to 500x500 (expected by the model)
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Ensure the input has the correct shape (1, 500, 500, 3)
    return np.expand_dims(image_np, axis=0)  # Add batch dimension

def postprocess_image(image_np, original_size):
    """Post-process the output image to displayable format."""
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    output_image = Image.fromarray(image_np)

    # Resize back to the original size
    return output_image.resize(original_size)

# Prediction and Display
if uploaded_file is not None:
    # Load and preprocess image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='📷 Uploaded Low Light Image', use_column_width=True)

    st.write("🔄 Enhancing image...")

    # Preprocess image
    processed_input = preprocess_image(input_image)

    # Make prediction
    prediction = model.predict(processed_input)

    # Squeeze the output to remove the batch dimension
    enhanced_img = prediction.squeeze()

    # Postprocess for display
    output_image = postprocess_image(enhanced_img, original_size=input_image.size)

    st.image(output_image, caption='✨ Enhanced Image', use_column_width=True)

    # Option to download the image
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(label="📥 Download Enhanced Image", data=byte_im, file_name="enhanced_image.png", mime="image/png")
