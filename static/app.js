(() => {
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const submitBtn = document.getElementById("submit-btn");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");

    const titleCase = (s) =>
        s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    const setStatus = (text, isError = false) => {
        statusEl.textContent = text;
        statusEl.classList.toggle("error", isError);
    };

    const renderResults = (data) => {
        const { top_breeds, top_confs, raw_b64, gradcam_b64, gradcam_failed } = data;

        const winnerBreed = titleCase(top_breeds[0]);
        const winnerPct = (top_confs[0] * 100).toFixed(1);

        const bars = top_breeds
            .map((breed, i) => {
                const pct = (top_confs[i] * 100).toFixed(1);
                return `
                    <div class="bar-row">
                        <div class="label">${titleCase(breed)}</div>
                        <div class="bar-track"><div class="bar-fill" style="width: ${pct}%"></div></div>
                        <div class="pct">${pct}%</div>
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
                Prediction: <strong>${winnerBreed}</strong>
                <span class="conf">(${winnerPct}%)</span>
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
            submitBtn.disabled = false;
        }
    });

    fileInput.addEventListener("change", () => {
        submitBtn.disabled = !(fileInput.files && fileInput.files[0]);
    });
})();
