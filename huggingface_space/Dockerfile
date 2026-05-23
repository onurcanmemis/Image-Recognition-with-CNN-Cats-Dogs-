# Pet Breed Classifier — Hugging Face Space (Docker SDK)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first so the slow TF download is cached across rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the Space contents: app.py, templates/, static/, the
# three model artefacts (cnn_model.keras, class_indices.json,
# model_info.json), and cats_and_dogs.jpg.
COPY . .

# Hugging Face Spaces routes external traffic to port 7860.
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
