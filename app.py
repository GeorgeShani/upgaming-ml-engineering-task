"""Local interactive pedestrian-detection demonstration."""

import json

import streamlit as st
import torch

from pedestrian.app_support import (
    DEFAULT_SELECTION,
    PROJECT_ROOT,
    decode_image,
    draw_predictions,
    load_selected_model,
    read_selection,
)
from pedestrian.inference import predict

st.set_page_config(page_title="Pedestrian detector", page_icon="🚶", layout="centered")


@st.cache_resource
def cached_model(selection_contents: str, checkpoint_size: int, checkpoint_mtime_ns: int):
    """Reload when the selection or the checkpoint file changes."""
    del selection_contents, checkpoint_size, checkpoint_mtime_ns
    torch.set_num_threads(4)
    return load_selected_model()


st.title("Pedestrian detector")
st.write("Upload a photo to find people and see their detection scores.")

try:
    selection_text = DEFAULT_SELECTION.read_text(encoding="utf-8")
    selection, checkpoint = read_selection()
    checkpoint_info = checkpoint.stat()
except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
    st.error(f"The selected model is unavailable: {error}")
    st.info("Follow the README to recreate the selected checkpoint, then reload this page.")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], max_upload_size=20)
sample_path = PROJECT_ROOT / "data" / "person1.jpg"
use_sample = st.checkbox("Use the example image", disabled=not sample_path.is_file())
threshold = st.slider("Minimum confidence", min_value=0.05, max_value=1.0, value=0.5, step=0.05)

data = (
    uploaded.getvalue()
    if uploaded is not None
    else sample_path.read_bytes()
    if use_sample
    else None
)
if data is None:
    st.info("Choose a JPEG or PNG image to begin.")
    st.stop()

try:
    image = decode_image(data)
    with st.spinner("Finding people…"):
        model, _ = cached_model(
            selection_text, checkpoint_info.st_size, checkpoint_info.st_mtime_ns
        )
        prediction = predict(image, model, threshold=threshold)
except (ValueError, OSError, RuntimeError, KeyError) as error:
    st.error(str(error))
    st.stop()

count = len(prediction["scores"])
st.metric("People detected", count)
st.image(
    draw_predictions(image, prediction), caption="Boxes show detections at the selected confidence."
)
if count:
    st.write(
        "Confidence scores: " + ", ".join(f"{score:.2f}" for score in prediction["scores"].tolist())
    )
else:
    st.info("No people were detected at this confidence level. Try lowering the threshold.")
