"""
frontend/app.py
----------------
Streamlit UI: upload an image, preview it, send it to the FastAPI
backend's /predict endpoint, and display the predicted label with a
probability breakdown, plus an optional metadata hint (informational
only, shown separately from the model's verdict).

Run:
    streamlit run frontend/app.py
"""

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="AI vs Real Image Detector", page_icon=":mag:")
st.title("AI-Generated vs Real Image Detector")
st.caption("Upload an image to check whether it looks real or AI-generated.")

uploaded_file = st.file_uploader(
    "Drag and drop an image, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Preview", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                response = requests.post(BACKEND_URL, files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    label = result["predicted_label"]

                    if label == "AI-Generated":
                        st.error(f"### Prediction: {label}")
                    else:
                        st.success(f"### Prediction: {label}")

                    col1, col2 = st.columns(2)
                    col1.metric("AI-Generated Probability", f"{result['ai_generated_probability']}%")
                    col2.metric("Real Probability", f"{result['real_probability']}%")

                    st.caption(result["disclaimer"])
                    st.caption(f"Model version: {result['model_version']}")

                    # Metadata hint — clearly separate from the model's verdict
                    meta = result.get("metadata_hint")
                    if meta and meta.get("matched_keywords"):
                        st.markdown("---")
                        st.info(
                            f"**Metadata hint (informational only):** found references to "
                            f"{', '.join(meta['matched_keywords'])} in the file's metadata.\n\n"
                            f"_{meta['note']}_"
                        )
                else:
                    error = response.json()
                    st.error(error.get("message", "Something went wrong"))

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the backend. Is it running? "
                    "(`uvicorn backend.main:app --port 8000`)"
                )
            except Exception:
                st.error("An unexpected error occurred.")
