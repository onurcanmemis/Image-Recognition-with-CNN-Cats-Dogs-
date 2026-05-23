---
title: Pet Breed Classifier
emoji: 🐾
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
license: mit
short_description: 37-class cat & dog breed classifier with Grad-CAM
---

# Pet Breed Classifier

Identifies one of 37 cat & dog breeds (Oxford-IIIT Pet) and shows a Grad-CAM
heatmap of the pixels that drove the prediction.

The model is the winner of a six-architecture tournament (baseline CNN,
deeper CNN, MobileNetV2 frozen/fine-tuned, EfficientNetB0 frozen/fine-tuned).
Tournament code, methodology, and results live in the source repo:

➡️ **GitHub:** https://huggingface.co/spaces/onurcanmemis/Dog_Breed_Classifier

## Use

1. Drop a JPG / PNG / JPEG of a single cat or dog into the upload box.
2. Press **Predict**.
3. Inspect the top-5 candidate breeds, the Grad-CAM heatmap, and the model
   card.

## How this Space is built

The Space uses the **Docker SDK**: the container is built from the
`Dockerfile` at the Space root, which installs the dependencies in
`requirements.txt` and runs `uvicorn app:app --host 0.0.0.0 --port 7860`.
The frontend is a hand-written HTML/CSS page served by FastAPI, not
Streamlit — there is no `sdk_version` to pin.

## Notes & limitations

- Trained on Oxford-IIIT Pet only. Out-of-distribution photos (animals
  outside the 37 breeds, multi-pet photos, very low-resolution images) will
  still produce a confident-looking top-1; treat below ~60 % confidence as
  unreliable.
- The Grad-CAM target layer is read from `model_info.json`; the same
  `app.py` works regardless of which architecture won the tournament.
