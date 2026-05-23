(() => {
    const form = document.getElementById("upload-form");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const filenameEl = document.getElementById("filename");
    const submitBtn = document.getElementById("submit-btn");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");

    const titleCase = (s) =>
        s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    const setStatus = (text, isError = false) => {
        statusEl.textContent = text;
        statusEl.classList.toggle("error", isError);
    };

    const syncFileState = () => {
        const file = fileInput.files && fileInput.files[0];
        filenameEl.textContent = file ? file.name : "";
        submitBtn.disabled = !file;
    };

    const renderResults = (data) => {
        const { top_breeds, top_confs, raw_b64, gradcam_b64, gradcam_failed } = data;

        const winnerBreed = titleCase(top_breeds[0]);
        const winnerPct = (top_confs[0] * 100).toFixed(1);
        const topConf = top_confs[0] || 1; // guard div-by-zero

        const bars = top_breeds
            .map((breed, i) => {
                const absPct = (top_confs[i] * 100).toFixed(1);
                const relPct = ((top_confs[i] / topConf) * 100).toFixed(1);
                return `
                    <div class="bar-row">
                        <div class="label">${titleCase(breed)}</div>
                        <div class="bar-track"><div class="bar-fill" style="width: ${relPct}%"></div></div>
                        <div class="pct">${absPct}%</div>
                    </div>`;
            })
            .join("");

        const camPanel = gradcam_b64
            ? `
                <figure>
                    <img src="data:image/png;base64,${raw_b64}" alt="Input (resized)" />
                    <figcaption>Input (resized)</figcaption>
                </figure>
                <figure>
                    <img src="data:image/png;base64,${gradcam_b64}" alt="Grad-CAM overlay" />
                    <figcaption>Grad-CAM overlay</figcaption>
                </figure>`
            : `
                <figure>
                    <img src="data:image/png;base64,${raw_b64}" alt="Input (resized)" />
                    <figcaption>Input (resized)</figcaption>
                </figure>
                <div class="warning">Grad-CAM unavailable: ${gradcam_failed || "unknown error"}</div>`;

        resultsEl.innerHTML = `
            <div class="headline">
                <span class="headline-label">Top prediction</span>
                <div class="headline-row">
                    <span class="headline-name">${winnerBreed}</span>
                    <span class="headline-conf">${winnerPct}%</span>
                </div>
            </div>
            <div class="results-grid">
                <section class="panel">
                    <h2>Top-5 candidates</h2>
                    <div class="bars">${bars}</div>
                </section>
                <section class="panel">
                    <h2>Grad-CAM</h2>
                    <div class="cam-grid">${camPanel}</div>
                </section>
            </div>
        `;
    };

    // Drag-and-drop wiring on the dropzone
    ["dragenter", "dragover"].forEach((evt) => {
        dropzone.addEventListener(evt, (event) => {
            event.preventDefault();
            event.stopPropagation();
            dropzone.classList.add("is-dragover");
        });
    });
    ["dragleave", "dragend"].forEach((evt) => {
        dropzone.addEventListener(evt, (event) => {
            event.preventDefault();
            event.stopPropagation();
            dropzone.classList.remove("is-dragover");
        });
    });
    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.remove("is-dragover");
        const files = event.dataTransfer && event.dataTransfer.files;
        if (!files || !files.length) return;
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        fileInput.files = dt.files;
        syncFileState();
    });

    fileInput.addEventListener("change", syncFileState);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const file = fileInput.files && fileInput.files[0];
        if (!file) {
            setStatus("Choose an image first.", true);
            return;
        }
        submitBtn.disabled = true;
        setStatus("Predicting…");
        resultsEl.innerHTML = "";

        try {
            const fd = new FormData();
            fd.append("file", file);
            const res = await fetch("/predict", { method: "POST", body: fd });
            if (!res.ok) {
                const body = await res.text();
                throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
            }
            const data = await res.json();
            renderResults(data);
            setStatus("");
        } catch (err) {
            setStatus(`Prediction failed: ${err.message}`, true);
        } finally {
            submitBtn.disabled = !(fileInput.files && fileInput.files[0]);
        }
    });
})();
