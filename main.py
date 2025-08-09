import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import time

st.set_page_config(page_title="AI Image Generator", layout="centered")

@st.cache_resource(show_spinner=True)
def load_model():
    """Load Stable Diffusion model only once and cache it."""
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32
    ).to("cpu")  
    pipe.enable_attention_slicing()
    return pipe

st.header("AI-Image Generator")

with st.spinner("Loading Stable Diffusion model... This may take a few minutes the first time."):
    pipe = load_model()

prompt = st.text_input(
    "Enter image prompt:",
    placeholder="e.g., 'A futuristic server room with glowing fiber-optic cables and AI cores'"
)

if st.button("Generate Image"):
    if not prompt.strip():
        st.error("Please enter a prompt before generating.")
    else:
        start_time = time.time()
        try:
            with st.spinner("Generating image... Please wait."):
                
                image = pipe(prompt, height=384, width=384).images[0]

                timestamp = int(time.time())
                filename = f"generated_image_{timestamp}.png"
                image.save(filename)

                st.image(image, caption=prompt, use_column_width=True)

                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    "Download Image",
                    data=buf.getvalue(),
                    file_name=filename,
                    mime="image/png"
                )

                st.success(f"Generated in {time.time() - start_time:.1f}s")
        except Exception as e:
            st.error(f"Generation interrupted: {str(e)}")
