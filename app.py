import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import io

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('image_enhancer_model.h5')
    return model

model = load_model()

# Title
st.title("Image Enhancer - Low Light to Clear Vision")

st.markdown("""
Upload a **low-light image**, and this app will enhance it using a trained deep learning model.

You can also adjust the **brightness** and **contrast** of the enhanced image.
""")

# Upload image
uploaded_file = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    """Resize and normalize image for model input."""
    image = image.convert('RGB')
    original_size = image.size  # Store original size
    resized = image.resize((500, 500))
    image_np = np.array(resized).astype(np.float32) / 255.0
    return np.expand_dims(image_np, axis=0), original_size

def postprocess_image(image_np, original_size):
    """Convert model output to image and resize back."""
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np).resize(original_size)
    return image

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='📷 Uploaded Low Light Image', use_column_width=True)

    st.write("🔄 Enhancing image...")

    processed_input, original_size = preprocess_image(input_image)
    prediction = model.predict(processed_input)
    enhanced_img_np = prediction.squeeze()

    output_image = postprocess_image(enhanced_img_np, original_size)

    st.subheader("🎛️ Adjust Enhanced Image")

    # Add sliders for brightness and contrast
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)

    enhancer_bright = ImageEnhance.Brightness(output_image)
    bright_img = enhancer_bright.enhance(brightness)

    enhancer_contrast = ImageEnhance.Contrast(bright_img)
    final_img = enhancer_contrast.enhance(contrast)

    st.image(final_img, caption='Enhanced Image (Adjusted)', use_column_width=True)

    # Download button
    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(label="📥 Download Enhanced Image", data=byte_im,
                       file_name="enhanced_image.png", mime="image/png")
