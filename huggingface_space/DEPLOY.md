# Deploying to Hugging Face Spaces (Docker SDK)

The Space is a separate git repository on `huggingface.co`. Treat the files
in this folder as **the manifest of what gets copied** into the Space repo
at deploy time — don't try to keep two `app.py` files in sync.

## One-time setup

1. Create a Space at <https://huggingface.co/new-space>:
   - SDK: **Docker** (not Streamlit — the previous deployment used the
     Streamlit SDK; that's been retired in favour of a FastAPI app served
     by uvicorn inside a container).
   - Hardware: CPU basic is fine for inference at 224×224.
   - Visibility: your call.
2. Clone the empty Space locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name> hf-space
   ```
3. `cnn_model.keras` is typically larger than 10 MB, which means the
   Space needs Git LFS. Inside the cloned Space repo, once, run:
   ```bash
   cd hf-space
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   git commit -m "Track .keras artefacts with LFS"
   ```

## Per-deploy

From the repo root:

```bash
# Application code + frontend assets
cp app.py                              hf-space/
cp -R templates                        hf-space/
cp -R static                           hf-space/

# Model artefacts
cp cnn_model.keras                     hf-space/
cp class_indices.json                  hf-space/
cp model_info.json                     hf-space/

# Container build files
cp huggingface_space/Dockerfile        hf-space/
cp huggingface_space/requirements.txt  hf-space/
cp huggingface_space/README.md         hf-space/README.md

cd hf-space
git add -A
git commit -m "Deploy: pet breed classifier $(date +%Y-%m-%d)"
git push
```

The Space rebuilds the Docker image automatically. First builds take
~3–5 min (TensorFlow is the bulk of the image); subsequent rebuilds with
the same `requirements.txt` reuse the pip layer and finish in under a
minute.

## What's in the bundle

| Source on disk                     | Lands at on the Space       | Purpose                            |
|------------------------------------|-----------------------------|------------------------------------|
| `app.py`                           | `app.py`                    | FastAPI app + Grad-CAM             |
| `templates/index.html`             | `templates/index.html`      | Upload page                        |
| `static/style.css`                 | `static/style.css`          | Dark theme                         |
| `static/app.js`                    | `static/app.js`             | Fetch + render results             |
| `static/cats_and_dogs.jpg`         | `static/cats_and_dogs.jpg`  | Banner image                       |
| `cnn_model.keras`                  | `cnn_model.keras`           | Winning model artefact             |
| `class_indices.json`               | `class_indices.json`        | Index → breed name                 |
| `model_info.json`                  | `model_info.json`           | Preprocessing tag, target layer    |
| `huggingface_space/Dockerfile`     | `Dockerfile`                | Container build recipe             |
| `huggingface_space/requirements.txt` | `requirements.txt`        | Python deps installed in the image |
| `huggingface_space/README.md`      | `README.md`                 | Space metadata (`sdk: docker`)     |

## Notes

- The Space listens on port **7860** — HF Spaces routes external traffic
  to that port. The Dockerfile's `EXPOSE 7860` + uvicorn's
  `--port 7860` cover this; the Space `README.md`'s `app_port: 7860`
  frontmatter declares it to HF.
- `requirements.txt` drops `tensorflow-metal` (Linux Space runners have
  no Apple GPUs) and uses `opencv-python-headless` instead of
  `opencv-python` (no GUI deps on a server).
- The `sdk_version` field that the Streamlit-SDK Space used is **not**
  applicable under the Docker SDK — the Python and library versions are
  fixed by the Dockerfile and `requirements.txt`.
- If you change `app.py`, `templates/`, or `static/` locally, the Space
  doesn't auto-sync — you must rerun the cp block + `git push` for the
  changes to take effect.
