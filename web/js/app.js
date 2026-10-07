/**
 * VERITAS Enterprise SOTA Deepfake Cyber-Forensics Studio
 * Frontend Application & Real-Time Evidentiary Visualizer
 */

document.addEventListener("DOMContentLoaded", () => {
    // Current analysis state
    let currentImageResult = null;
    let currentVideoResult = null;
    let currentAudioResult = null;
    let currentActiveView = "hud"; // "hud", "gradcam", "original"

    // Live Interceptor state
    let liveStream = null;
    let liveInterval = null;
    let liveFrameCount = 0;

    // -------------------------------------------------------------
    // 1. Tab Navigation
    // -------------------------------------------------------------
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const pane = document.getElementById(targetTab);
            if (pane) pane.classList.add("active");

            // Dynamic view refreshes on tab activation
            if (targetTab === "tab-spectral" && currentImageResult && currentImageResult.spectral_metrics) {
                renderAzimuthalChart(currentImageResult.spectral_metrics.radial_profile, currentImageResult.spectral_metrics.slope);
            } else if (targetTab === "tab-cases") {
                loadCasesList();
            } else if (targetTab === "tab-siem") {
                loadSIEMConfig();
            }
        });
    });

    // -------------------------------------------------------------
    // 2. Initial System Telemetry
    // -------------------------------------------------------------
    async function checkTelemetry() {
        try {
            const res = await fetch("/api/telemetry");
            if (res.ok) {
                const data = await res.json();
                const siemEl = document.getElementById("telemetry-siem-status");
                if (siemEl && data.siem_status) {
                    siemEl.innerText = data.siem_status.enabled ? "Active Webhook" : "Standby";
                    siemEl.className = data.siem_status.enabled ? "t-value text-emerald" : "t-value text-amber";
                }
            }
        } catch (e) {
            console.warn("Telemetry fetch error:", e);
        }
    }
    checkTelemetry();

    // -------------------------------------------------------------
    // 3. Image Forensics Studio
    // -------------------------------------------------------------
    const imageDropzone = document.getElementById("image-dropzone");
    const imageInput = document.getElementById("image-input");
    const imageLoading = document.getElementById("image-loading");
    const imageResultsView = document.getElementById("image-results-view");

    if (imageDropzone) {
        imageDropzone.addEventListener("click", () => imageInput.click());
        imageDropzone.addEventListener("dragover", (e) => { e.preventDefault(); imageDropzone.classList.add("dragover"); });
        imageDropzone.addEventListener("dragleave", () => imageDropzone.classList.remove("dragover"));
        imageDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            imageDropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleImageUpload(e.dataTransfer.files[0]);
            }
        });
    }

    if (imageInput) {
        imageInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleImageUpload(e.target.files[0]);
            }
        });
    }

    const btnLoadReal = document.getElementById("btn-load-real");
    const btnLoadFake = document.getElementById("btn-load-fake");
    if (btnLoadReal) btnLoadReal.addEventListener("click", () => fetchImageAnalysis({ sample_name: "test_authentic.jpg" }));
    if (btnLoadFake) btnLoadFake.addEventListener("click", () => fetchImageAnalysis({ sample_name: "test_synthetic.jpg" }));

    function handleImageUpload(file) {
        const formData = new FormData();
        formData.append("file", file);
        fetchImageAnalysis(formData);
    }

    async function fetchImageAnalysis(payload) {
        if (imageLoading) imageLoading.style.display = "block";
        if (imageResultsView) imageResultsView.style.display = "none";

        try {
            let res;
            if (payload instanceof FormData) {
                res = await fetch("/api/analyze/image", { method: "POST", body: payload });
            } else {
                const fd = new FormData();
                fd.append("sample_name", payload.sample_name);
                res = await fetch("/api/analyze/image", { method: "POST", body: fd });
            }

            if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
            const data = await res.json();
            currentImageResult = data;
            renderImageResults(data);
        } catch (err) {
            alert(`Image analysis error: ${err.message}`);
        } finally {
            if (imageLoading) imageLoading.style.display = "none";
        }
    }

    function renderImageResults(data) {
        if (!imageResultsView) return;
        imageResultsView.style.display = "block";

        // Verdict banner
        const banner = document.getElementById("verdict-banner");
        if (banner) {
            banner.className = "verdict-banner";
            if (data.severity === "CRITICAL") banner.classList.add("banner-critical");
            else if (data.severity === "CLEAN") banner.classList.add("banner-clean");
            else banner.classList.add("banner-warning");
        }

        const titleEl = document.getElementById("verdict-title");
        if (titleEl) {
            titleEl.innerText = data.verdict;
            titleEl.style.color = data.badge_color;
        }

        const confEl = document.getElementById("confidence-val");
        if (confEl) confEl.innerText = `${data.confidence.toFixed(1)}%`;

        const elapsedEl = document.getElementById("meta-elapsed");
        if (elapsedEl) elapsedEl.innerText = `${data.elapsed_ms}ms`;

        const facesEl = document.getElementById("meta-faces");
        if (facesEl) facesEl.innerText = `${data.faces_detected} Primary Subject`;

        const shaEl = document.getElementById("meta-sha256");
        if (shaEl) shaEl.innerText = data.sha256 ? `${data.sha256.substring(0, 16)}...` : "N/A";

        // Main Visualizer
        currentActiveView = "hud";
        updateMainDisplay();

        // Multi-Cue Breakdown
        const cues = data.cue_breakdown || {};
        setBar("bar-spatial", "score-spatial", cues["Spatial Deep Learning"] || 0);
        setBar("bar-fft", "score-fft", cues["Frequency Domain (FFT)"] || 0);
        setBar("bar-ela", "score-ela", cues["Error Level Analysis (ELA)"] || 0);
        setBar("bar-boundary", "score-boundary", cues["Face-Background Boundary"] || 0);

        // Generator Attribution
        renderGeneratorAttribution(data.generator_attribution);

        // Anomalies Checklist
        const anomaliesList = document.getElementById("anomalies-list");
        if (anomaliesList && data.anomalies) {
            anomaliesList.innerHTML = "";
            data.anomalies.forEach(anom => {
                const li = document.createElement("li");
                li.innerText = anom;
                anomaliesList.appendChild(li);
            });
        }

        // Evidence Plates
        const pFace = document.getElementById("plate-face");
        const pFft = document.getElementById("plate-fft");
        const pEla = document.getElementById("plate-ela");
        if (pFace && data.images.face_crop) pFace.src = data.images.face_crop;
        if (pFft && data.images.fft_spectrum) pFft.src = data.images.fft_spectrum;
        if (pEla && data.images.ela_heatmap) pEla.src = data.images.ela_heatmap;

        const fftMeta = document.getElementById("fft-meta");
        if (fftMeta && data.spectral_metrics) {
            fftMeta.innerText = `Radial Decay Slope: ${data.spectral_metrics.slope} • HFER: ${data.spectral_metrics.hfer}`;
        }

        const elaMeta = document.getElementById("ela-meta");
        if (elaMeta && data.ela_metrics) {
            elaMeta.innerText = `Face-to-Background Compression Disparity: ${(data.ela_metrics.disparity_ratio * 100).toFixed(1)}%`;
        }

        imageResultsView.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderGeneratorAttribution(attr) {
        if (!attr) return;
        const topLabel = document.getElementById("attr-top-label");
        if (topLabel) {
            topLabel.innerText = `${attr.primary_attribution} (${attr.attribution_confidence.toFixed(1)}%)`;
        }

        const barsContainer = document.getElementById("gen-attr-bars");
        if (!barsContainer || !attr.distribution) return;

        barsContainer.innerHTML = "";
        Object.entries(attr.distribution).forEach(([genName, pct]) => {
            const row = document.createElement("div");
            row.className = "gen-attr-row";
            row.innerHTML = `
                <div class="gen-attr-meta">
                    <span style="color:var(--text-secondary);">${genName}</span>
                    <span class="mono" style="color:#38bdf8;">${pct.toFixed(1)}%</span>
                </div>
                <div class="gen-attr-bar-bg">
                    <div class="gen-attr-fill" style="width:${Math.max(2, pct)}%;"></div>
                </div>
            `;
            barsContainer.appendChild(row);
        });
    }

    function setBar(barId, labelId, value) {
        const bar = document.getElementById(barId);
        const label = document.getElementById(labelId);
        if (!bar || !label) return;
        const clamped = Math.max(0, Math.min(100, value));
        label.innerText = `${clamped.toFixed(1)}%`;
        bar.style.width = `${clamped}%`;
        if (clamped >= 70) bar.style.background = "#ef4444";
        else if (clamped >= 40) bar.style.background = "#f59e0b";
        else bar.style.background = "#10b981";
    }

    // View Toggles (HUD vs Grad-CAM vs Raw)
    const viewButtons = document.querySelectorAll(".view-toggles .btn-toggle");
    viewButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            viewButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentActiveView = btn.getAttribute("data-view");
            updateMainDisplay();
        });
    });

    const camSlider = document.getElementById("cam-opacity");
    const camSliderContainer = document.getElementById("gradcam-slider-container");
    const camOpacityVal = document.getElementById("cam-opacity-val");

    if (camSlider) {
        camSlider.addEventListener("input", (e) => {
            if (camOpacityVal) camOpacityVal.innerText = `${e.target.value}%`;
            const mainImg = document.getElementById("main-forensic-image");
            if (mainImg && currentActiveView === "gradcam") {
                mainImg.style.opacity = (e.target.value / 100).toString();
            }
        });
    }

    function updateMainDisplay() {
        const mainImg = document.getElementById("main-forensic-image");
        if (!mainImg || !currentImageResult) return;

        if (currentActiveView === "hud") {
            mainImg.src = currentImageResult.images.hud;
            mainImg.style.opacity = "1";
            if (camSliderContainer) camSliderContainer.style.display = "none";
        } else if (currentActiveView === "gradcam") {
            mainImg.src = currentImageResult.images.gradcam_overlay;
            if (camSliderContainer) {
                camSliderContainer.style.display = "flex";
                const op = camSlider ? camSlider.value / 100 : 0.75;
                mainImg.style.opacity = op.toString();
            }
        } else {
            mainImg.src = currentImageResult.images.face_crop;
            mainImg.style.opacity = "1";
            if (camSliderContainer) camSliderContainer.style.display = "none";
        }
    }

    // Export PDF & JSON
    const btnExportPdf = document.getElementById("btn-export-pdf");
    const btnExportJson = document.getElementById("btn-export-json");

    if (btnExportPdf) {
        btnExportPdf.addEventListener("click", async () => {
            if (!currentImageResult) return;
            try {
                const res = await fetch("/api/export/pdf", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: jsonStringifySafe({ media_name: currentImageResult.media_name, forensic_result: currentImageResult })
                });
                if (res.ok) {
                    const blob = await res.blob();
                    downloadBlob(blob, `VERITAS_Certificate_${currentImageResult.media_name}.pdf`);
                }
            } catch (err) { alert(`PDF export error: ${err.message}`); }
        });
    }

    if (btnExportJson) {
        btnExportJson.addEventListener("click", async () => {
            if (!currentImageResult) return;
            try {
                const res = await fetch("/api/export/json", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: jsonStringifySafe({ media_name: currentImageResult.media_name, forensic_result: currentImageResult })
                });
                if (res.ok) {
                    const blob = await res.blob();
                    downloadBlob(blob, `VERITAS_Dossier_${currentImageResult.media_name}.json`);
                }
            } catch (err) { alert(`JSON export error: ${err.message}`); }
        });
    }

    // Attach to active case buttons
    const btnAttachImageCase = document.getElementById("btn-attach-image-case");
    if (btnAttachImageCase) {
        btnAttachImageCase.addEventListener("click", () => attachCurrentExhibitToCase(currentImageResult));
    }

    const btnAttachVideoCase = document.getElementById("btn-attach-video-case");
    if (btnAttachVideoCase) {
        btnAttachVideoCase.addEventListener("click", () => attachCurrentExhibitToCase(currentVideoResult));
    }

    const btnAttachAudioCase = document.getElementById("btn-attach-audio-case");
    if (btnAttachAudioCase) {
        btnAttachAudioCase.addEventListener("click", () => attachCurrentExhibitToCase(currentAudioResult));
    }

    // -------------------------------------------------------------
    // 4. Video Temporal Stream Forensics
    // -------------------------------------------------------------
    const videoDropzone = document.getElementById("video-dropzone");
    const videoInput = document.getElementById("video-input");
    const videoLoading = document.getElementById("video-loading");
    const videoResultsView = document.getElementById("video-results-view");

    if (videoDropzone) {
        videoDropzone.addEventListener("click", () => videoInput.click());
        videoDropzone.addEventListener("dragover", (e) => { e.preventDefault(); videoDropzone.classList.add("dragover"); });
        videoDropzone.addEventListener("dragleave", () => videoDropzone.classList.remove("dragover"));
        videoDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            videoDropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleVideoUpload(e.dataTransfer.files[0]);
            }
        });
    }

    if (videoInput) {
        videoInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleVideoUpload(e.target.files[0]);
            }
        });
    }

    async function handleVideoUpload(file) {
        if (videoLoading) videoLoading.style.display = "block";
        if (videoResultsView) videoResultsView.style.display = "none";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/analyze/video", { method: "POST", body: formData });
            if (!res.ok) throw new Error(`Video analysis failed (HTTP ${res.status})`);
            const data = await res.json();
            currentVideoResult = data;
            renderVideoResults(data);
        } catch (err) {
            alert(`Video analysis error: ${err.message}`);
        } finally {
            if (videoLoading) videoLoading.style.display = "none";
        }
    }

    function renderVideoResults(data) {
        if (!videoResultsView) return;
        videoResultsView.style.display = "block";

        const banner = document.getElementById("video-verdict-banner");
        if (banner) {
            banner.className = "verdict-banner";
            if (data.severity === "CRITICAL") banner.classList.add("banner-critical");
            else if (data.severity === "CLEAN") banner.classList.add("banner-clean");
            else banner.classList.add("banner-warning");
        }

        const titleEl = document.getElementById("video-verdict-title");
        if (titleEl) {
            titleEl.innerText = data.verdict;
            titleEl.style.color = data.badge_color;
        }

        const confEl = document.getElementById("video-confidence-val");
        if (confEl) confEl.innerText = `${data.confidence.toFixed(1)}%`;

        const framesEl = document.getElementById("video-meta-frames");
        if (framesEl) framesEl.innerText = `${data.total_frames_analyzed} Keyframes`;

        const shaEl = document.getElementById("video-meta-sha256");
        if (shaEl) shaEl.innerText = data.sha256 ? `${data.sha256.substring(0, 16)}...` : "N/A";

        // Temporal Metrics
        const blinkEl = document.getElementById("metric-blink-rate");
        if (blinkEl && data.temporal_metrics) blinkEl.innerText = `${data.temporal_metrics.blink_frequency} /min`;

        const jitterEl = document.getElementById("metric-motion-jitter");
        if (jitterEl && data.temporal_metrics) jitterEl.innerText = `${data.temporal_metrics.mean_optical_flow_jitter} px`;

        const fakeFramesEl = document.getElementById("metric-fake-frames");
        if (fakeFramesEl) fakeFramesEl.innerText = `${data.fake_frame_count} / ${data.total_frames_analyzed}`;

        const bioVerdictEl = document.getElementById("metric-bio-verdict");
        if (bioVerdictEl && data.temporal_metrics) bioVerdictEl.innerText = data.temporal_metrics.temporal_verdict;

        // Render Timeline Canvas
        if (data.timeline) renderVideoTimeline(data.timeline);

        videoResultsView.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderVideoTimeline(timeline) {
        const canvas = document.getElementById("video-timeline-canvas");
        if (!canvas || !timeline || timeline.length === 0) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Danger threshold line (50%)
        const thresholdY = h - 30 - 0.5 * (h - 60);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(50, thresholdY);
        ctx.lineTo(w - 20, thresholdY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#ef4444";
        ctx.font = "10px JetBrains Mono";
        ctx.fillText("50% Suspicious Threshold", w - 170, thresholdY - 6);

        // Curve line
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        const step = (w - 70) / Math.max(1, timeline.length - 1);

        timeline.forEach((pt, idx) => {
            const x = 50 + idx * step;
            const y = h - 30 - (pt.fake_prob / 100) * (h - 60);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Data points
        timeline.forEach((pt, idx) => {
            const x = 50 + idx * step;
            const y = h - 30 - (pt.fake_prob / 100) * (h - 60);
            ctx.fillStyle = pt.fake_prob >= 50 ? "#ef4444" : "#10b981";
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // -------------------------------------------------------------
    // 5. Audio & Voice Clone Forensics Studio
    // -------------------------------------------------------------
    const audioDropzone = document.getElementById("audio-dropzone");
    const audioInput = document.getElementById("audio-input");
    const audioLoading = document.getElementById("audio-loading");
    const audioResultsView = document.getElementById("audio-results-view");

    if (audioDropzone) {
        audioDropzone.addEventListener("click", () => audioInput.click());
        audioDropzone.addEventListener("dragover", (e) => { e.preventDefault(); audioDropzone.classList.add("dragover"); });
        audioDropzone.addEventListener("dragleave", () => audioDropzone.classList.remove("dragover"));
        audioDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            audioDropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleAudioUpload(e.dataTransfer.files[0]);
            }
        });
    }

    if (audioInput) {
        audioInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleAudioUpload(e.target.files[0]);
            }
        });
    }

    async function handleAudioUpload(file) {
        if (audioLoading) audioLoading.style.display = "block";
        if (audioResultsView) audioResultsView.style.display = "none";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/analyze/audio", { method: "POST", body: formData });
            if (!res.ok) throw new Error(`Audio analysis failed (HTTP ${res.status})`);
            const data = await res.json();
            currentAudioResult = data;
            renderAudioResults(data);
        } catch (err) {
            alert(`Audio analysis error: ${err.message}`);
        } finally {
            if (audioLoading) audioLoading.style.display = "none";
        }
    }

    function renderAudioResults(data) {
        if (!audioResultsView) return;
        audioResultsView.style.display = "block";

        const banner = document.getElementById("audio-verdict-banner");
        if (banner) {
            banner.className = "verdict-banner";
            if (data.severity === "HIGH") banner.classList.add("banner-critical");
            else if (data.severity === "LOW") banner.classList.add("banner-clean");
            else banner.classList.add("banner-warning");
        }

        const titleEl = document.getElementById("audio-verdict-title");
        if (titleEl) {
            titleEl.innerText = data.verdict;
            titleEl.style.color = data.badge_color;
        }

        const confEl = document.getElementById("audio-confidence-val");
        if (confEl) confEl.innerText = `${data.confidence.toFixed(1)}%`;

        const durEl = document.getElementById("audio-meta-duration");
        if (durEl) durEl.innerText = `${data.duration_seconds}s`;

        const srEl = document.getElementById("audio-meta-sr");
        if (srEl) srEl.innerText = `${data.sample_rate || 16000} Hz`;

        const shaEl = document.getElementById("audio-meta-sha256");
        if (shaEl) shaEl.innerText = data.sha256 ? `${data.sha256.substring(0, 16)}...` : "N/A";

        // Spectrogram Image
        const specImg = document.getElementById("audio-spectrogram-img");
        if (specImg && data.spectrogram_image) specImg.src = data.spectrogram_image;

        // Acoustic Cues
        const metrics = data.metrics || {};
        const hfVal = Math.min(100, (metrics.hf_energy_ratio || 0) * 100);
        setBar("bar-hf-ratio", "score-hf-ratio", hfVal);

        const flatVal = Math.min(100, (metrics.spectral_flatness || 0) * 500);
        setBar("bar-flatness", "score-flatness", flatVal);

        const jitterVal = Math.min(100, (metrics.pitch_jitter_index || 0) * 40);
        setBar("bar-jitter", "score-jitter", jitterVal);

        // Anomalies
        const listEl = document.getElementById("audio-anomalies-list");
        if (listEl && data.anomalies) {
            listEl.innerHTML = "";
            data.anomalies.forEach(anom => {
                const li = document.createElement("li");
                li.innerText = anom;
                listEl.appendChild(li);
            });
        }

        audioResultsView.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // -------------------------------------------------------------
    // 6. Live KYC & Webcam Interceptor
    // -------------------------------------------------------------
    const btnStartLive = document.getElementById("btn-start-live");
    const btnStopLive = document.getElementById("btn-stop-live");
    const liveVideo = document.getElementById("live-video");
    const liveCanvas = document.getElementById("live-canvas");
    const liveStandby = document.getElementById("live-standby-overlay");
    const liveStatusPill = document.getElementById("live-status-pill");

    if (btnStartLive) {
        btnStartLive.addEventListener("click", async () => {
            try {
                liveStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 } }
                });
                liveVideo.srcObject = liveStream;
                await liveVideo.play();

                if (liveStandby) liveStandby.style.display = "none";
                if (btnStopLive) btnStopLive.style.display = "block";
                if (liveStatusPill) {
                    liveStatusPill.innerText = "DEFENSE ACTIVE";
                    liveStatusPill.className = "badge-pill text-emerald";
                }

                liveFrameCount = 0;
                liveInterval = setInterval(processLiveInterceptorFrame, 500);
            } catch (err) {
                alert(`Webcam access denied or unavailable: ${err.message}`);
            }
        });
    }

    if (btnStopLive) {
        btnStopLive.addEventListener("click", stopLiveInterceptor);
    }

    function stopLiveInterceptor() {
        if (liveInterval) clearInterval(liveInterval);
        if (liveStream) {
            liveStream.getTracks().forEach(t => t.stop());
            liveStream = null;
        }
        if (liveStandby) liveStandby.style.display = "flex";
        if (btnStopLive) btnStopLive.style.display = "none";
        if (liveStatusPill) {
            liveStatusPill.innerText = "STANDBY";
            liveStatusPill.className = "badge-pill";
        }
    }

    async function processLiveInterceptorFrame() {
        if (!liveVideo || liveVideo.paused || liveVideo.ended) return;
        const ctx = liveCanvas.getContext("2d");
        const w = liveCanvas.width;
        const h = liveCanvas.height;

        // Draw current camera frame to canvas
        ctx.drawImage(liveVideo, 0, 0, w, h);
        const frameB64 = liveCanvas.toDataURL("image/jpeg", 0.75);

        liveFrameCount++;
        const frameCountEl = document.getElementById("live-frames-count");
        if (frameCountEl) frameCountEl.innerText = liveFrameCount.toString();

        try {
            const res = await fetch("/api/analyze/stream-frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame_b64: frameB64 })
            });

            if (res.ok) {
                const data = await res.json();
                updateLiveHUD(data, ctx, w, h);
            }
        } catch (e) {
            console.warn("Live frame drop:", e);
        }
    }

    function updateLiveHUD(data, ctx, w, h) {
        // Redraw video frame
        ctx.drawImage(liveVideo, 0, 0, w, h);

        const verdictEl = document.getElementById("live-verdict-display");
        const threatStatusEl = document.getElementById("live-threat-status");
        const riskEl = document.getElementById("live-risk-val");
        const livenessEl = document.getElementById("live-liveness-val");
        const latencyEl = document.getElementById("live-latency-val");

        if (latencyEl) latencyEl.innerText = `${data.latency_ms} ms`;

        if (!data.face_detected) {
            if (threatStatusEl) threatStatusEl.innerText = "No Face in View";
            if (riskEl) riskEl.innerText = "0.0%";
            if (livenessEl) livenessEl.innerText = "0.0%";
            return;
        }

        if (verdictEl) {
            verdictEl.innerText = data.verdict;
            verdictEl.style.color = data.badge_color;
        }
        if (threatStatusEl) threatStatusEl.innerText = data.threat_status;
        if (riskEl) riskEl.innerText = `${data.fake_probability}%`;
        if (livenessEl) livenessEl.innerText = `${data.liveness_score}%`;

        // Draw tactical HUD bounding box & corner brackets
        const [bx, by, bw, bh] = data.box;
        ctx.strokeStyle = data.badge_color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, bw, bh);

        // Corner brackets
        const cornerLen = Math.min(20, bw / 4);
        ctx.lineWidth = 3;
        ctx.beginPath();
        // Top-left
        ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
        // Top-right
        ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
        // Bottom-left
        ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
        // Bottom-right
        ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
        ctx.stroke();

        // Label Tag
        ctx.fillStyle = data.badge_color;
        ctx.fillRect(bx, by - 24, bw, 24);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px JetBrains Mono";
        ctx.fillText(`${data.threat_status} (${data.fake_probability}%)`, bx + 6, by - 7);
    }

    // -------------------------------------------------------------
    // 7. Case Vault & Legal Custody
    // -------------------------------------------------------------
    let allCases = [];
    let activeCaseId = null;

    const caseSelector = document.getElementById("case-selector");
    const btnNewCase = document.getElementById("btn-new-case");
    const modalNewCase = document.getElementById("modal-new-case");
    const btnCloseCaseModal = document.getElementById("btn-close-case-modal");
    const btnCancelCase = document.getElementById("btn-cancel-case");
    const formNewCase = document.getElementById("form-new-case");
    const btnExportMasterPdf = document.getElementById("btn-export-master-pdf");

    if (caseSelector) {
        caseSelector.addEventListener("change", (e) => {
            activeCaseId = e.target.value;
            renderActiveCaseDetails();
        });
    }

    if (btnNewCase && modalNewCase) {
        btnNewCase.addEventListener("click", () => modalNewCase.style.display = "flex");
    }

    if (btnCloseCaseModal && modalNewCase) {
        btnCloseCaseModal.addEventListener("click", () => modalNewCase.style.display = "none");
    }

    if (btnCancelCase && modalNewCase) {
        btnCancelCase.addEventListener("click", () => modalNewCase.style.display = "none");
    }

    if (formNewCase) {
        formNewCase.addEventListener("submit", async (e) => {
            e.preventDefault();
            const title = document.getElementById("input-case-title").value;
            const investigator = document.getElementById("input-case-investigator").value;
            const org = document.getElementById("input-case-org").value;

            try {
                const res = await fetch("/api/cases", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ title, investigator, organization: org })
                });
                if (res.ok) {
                    const newCase = await res.json();
                    modalNewCase.style.display = "none";
                    formNewCase.reset();
                    await loadCasesList();
                    caseSelector.value = newCase.case_id;
                    activeCaseId = newCase.case_id;
                    renderActiveCaseDetails();
                }
            } catch (err) {
                alert(`Error creating case: ${err.message}`);
            }
        });
    }

    if (btnExportMasterPdf) {
        btnExportMasterPdf.addEventListener("click", async () => {
            if (!activeCaseId) {
                alert("Please select or open an active case first.");
                return;
            }
            try {
                const res = await fetch(`/api/cases/${activeCaseId}/export/pdf`);
                if (res.ok) {
                    const blob = await res.blob();
                    downloadBlob(blob, `VERITAS_Master_Judicial_Dossier_${activeCaseId}.pdf`);
                } else {
                    alert("Master PDF dossier generation failed.");
                }
            } catch (err) {
                alert(`Export error: ${err.message}`);
            }
        });
    }

    async function loadCasesList() {
        try {
            const res = await fetch("/api/cases");
            if (res.ok) {
                const data = await res.json();
                allCases = data.cases || [];
                if (caseSelector) {
                    caseSelector.innerHTML = "";
                    allCases.forEach(c => {
                        const opt = document.createElement("option");
                        opt.value = c.case_id;
                        opt.innerText = `${c.case_id} — ${c.title} (${c.exhibits ? c.exhibits.length : 0} exhibits)`;
                        caseSelector.appendChild(opt);
                    });
                    if (allCases.length > 0 && !activeCaseId) {
                        activeCaseId = allCases[0].case_id;
                        caseSelector.value = activeCaseId;
                    }
                }
                renderActiveCaseDetails();
            }
        } catch (e) {
            console.warn("Failed to load cases:", e);
        }
    }

    function renderActiveCaseDetails() {
        const activeCase = allCases.find(c => c.case_id === activeCaseId);
        if (!activeCase) return;

        const idBadge = document.getElementById("case-id-badge");
        const titleDisp = document.getElementById("case-title-display");
        const agencyDisp = document.getElementById("case-agency-display");
        const countDisp = document.getElementById("case-exhibits-count");
        const tbody = document.getElementById("case-exhibits-tbody");

        if (idBadge) idBadge.innerText = activeCase.case_id;
        if (titleDisp) titleDisp.innerText = activeCase.title;
        if (agencyDisp) agencyDisp.innerText = `Lead: ${activeCase.investigator} • Agency: ${activeCase.organization} • Opened: ${activeCase.created_utc.split("T")[0]}`;
        if (countDisp) countDisp.innerText = `${activeCase.exhibits ? activeCase.exhibits.length : 0} Exhibits Attached`;

        if (tbody) {
            tbody.innerHTML = "";
            if (!activeCase.exhibits || activeCase.exhibits.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:24px;">No exhibits currently attached to this case. Use "Attach to Active Case" in any forensic lab.</td></tr>`;
            } else {
                activeCase.exhibits.forEach(ex => {
                    const row = document.createElement("tr");
                    const isFake = ex.verdict && ex.verdict.toLowerCase().includes("synthetic");
                    const color = isFake ? "#ef4444" : "#10b981";
                    row.innerHTML = `
                        <td class="mono" style="color:#38bdf8;">${ex.exhibit_id || "-"}</td>
                        <td><strong>${ex.media_name}</strong></td>
                        <td style="color:${color}; font-weight:700;">${ex.verdict}</td>
                        <td class="mono">${(ex.confidence || 0).toFixed(1)}%</td>
                        <td class="mono" style="font-size:0.75rem; color:var(--text-muted);">${ex.sha256 ? ex.sha256.substring(0, 16) + '...' : '-'}</td>
                        <td style="font-size:0.75rem; color:var(--text-secondary);">${ex.attached_utc ? ex.attached_utc.replace("T", " ").substring(0, 19) : "-"}</td>
                    `;
                    tbody.appendChild(row);
                });
            }
        }
    }

    async function attachCurrentExhibitToCase(resultObj) {
        if (!resultObj) {
            alert("Please run an analysis first before attaching to a case.");
            return;
        }
        if (!activeCaseId) {
            await loadCasesList();
            if (!activeCaseId) {
                alert("No active case exists. Please open a case in the Case Vault tab first.");
                return;
            }
        }

        const payload = {
            media_name: resultObj.media_name || "evidence_item",
            verdict: resultObj.verdict || "UNKNOWN",
            confidence: resultObj.confidence || 0.0,
            sha256: resultObj.sha256 || "UNKNOWN",
            threat_actor: resultObj.generator_attribution ? resultObj.generator_attribution.primary_attribution : "Unknown",
            notes: "Filed via VERITAS Cyber-Forensic Web Studio."
        };

        try {
            const res = await fetch(`/api/cases/${activeCaseId}/exhibits`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                alert(`Successfully attached exhibit ${payload.media_name} to case ${activeCaseId}!`);
                loadCasesList();
            } else {
                alert("Failed to attach exhibit to case.");
            }
        } catch (err) {
            alert(`Error attaching exhibit: ${err.message}`);
        }
    }

    // -------------------------------------------------------------
    // 8. SIEM & SOC Alert Integration
    // -------------------------------------------------------------
    const formSiemConfig = document.getElementById("form-siem-config");
    const siemEnabled = document.getElementById("siem-enabled");
    const siemWebhookUrl = document.getElementById("siem-webhook-url");
    const siemThreshold = document.getElementById("siem-threshold");
    const siemThresholdLabel = document.getElementById("siem-threshold-label");
    const siemHmacSecret = document.getElementById("siem-hmac-secret");
    const btnTestSiemPing = document.getElementById("btn-test-siem-ping");
    const siemStatusMsg = document.getElementById("siem-status-message");

    if (siemThreshold && siemThresholdLabel) {
        siemThreshold.addEventListener("input", (e) => {
            siemThresholdLabel.innerText = `${e.target.value}%`;
        });
    }

    async function loadSIEMConfig() {
        try {
            const res = await fetch("/api/siem/config");
            if (res.ok) {
                const cfg = await res.json();
                if (siemEnabled) siemEnabled.checked = !!cfg.enabled;
                if (siemWebhookUrl) siemWebhookUrl.value = cfg.webhook_url || "";
                if (siemThreshold) {
                    siemThreshold.value = Math.round((cfg.alert_threshold || 0.7) * 100);
                    if (siemThresholdLabel) siemThresholdLabel.innerText = `${siemThreshold.value}%`;
                }
            }
        } catch (e) {
            console.warn("Failed to load SIEM config:", e);
        }
    }

    if (formSiemConfig) {
        formSiemConfig.addEventListener("submit", async (e) => {
            e.preventDefault();
            const payload = {
                enabled: siemEnabled.checked,
                webhook_url: siemWebhookUrl.value,
                alert_threshold: parseInt(siemThreshold.value) / 100.0,
                hmac_secret: siemHmacSecret.value || ""
            };

            try {
                const res = await fetch("/api/siem/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    if (siemStatusMsg) {
                        siemStatusMsg.innerText = "SIEM webhook settings saved successfully.";
                        siemStatusMsg.style.color = "#10b981";
                    }
                    checkTelemetry();
                }
            } catch (err) {
                if (siemStatusMsg) {
                    siemStatusMsg.innerText = `Save error: ${err.message}`;
                    siemStatusMsg.style.color = "#ef4444";
                }
            }
        });
    }

    if (btnTestSiemPing) {
        btnTestSiemPing.addEventListener("click", async () => {
            if (siemStatusMsg) {
                siemStatusMsg.innerText = "Dispatching canary alert ping...";
                siemStatusMsg.style.color = "#38bdf8";
            }
            try {
                const res = await fetch("/api/siem/test", { method: "POST" });
                const data = await res.json();
                if (data.dispatched && data.delivery_record && data.delivery_record.status === "DELIVERED") {
                    siemStatusMsg.innerText = `Canary delivered successfully! (HTTP ${data.delivery_record.http_status})`;
                    siemStatusMsg.style.color = "#10b981";
                } else {
                    const reason = data.reason || (data.delivery_record ? data.delivery_record.error : "Unknown delivery error");
                    siemStatusMsg.innerText = `Dispatch report: ${reason}`;
                    siemStatusMsg.style.color = "#f59e0b";
                }
            } catch (err) {
                if (siemStatusMsg) {
                    siemStatusMsg.innerText = `Canary test error: ${err.message}`;
                    siemStatusMsg.style.color = "#ef4444";
                }
            }
        });
    }

    // -------------------------------------------------------------
    // 9. Deep Spectral Canvas (1/f Profile)
    // -------------------------------------------------------------
    function renderAzimuthalChart(radialProfile, slope) {
        const canvas = document.getElementById("azimuthal-canvas");
        if (!canvas || !radialProfile) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        const padL = 40, padR = 20, padT = 30, padB = 40;
        const plotW = w - padL - padR;
        const plotH = h - padT - padB;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, padT);
        ctx.lineTo(padL, h - padB);
        ctx.lineTo(w - padR, h - padB);
        ctx.stroke();

        const n = radialProfile.length;
        const maxVal = Math.max(...radialProfile, 1);
        const minVal = Math.min(...radialProfile, 0);
        const range = maxVal - minVal || 1;

        ctx.strokeStyle = "#00f5ff";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        radialProfile.forEach((val, i) => {
            const x = padL + (i / n) * plotW;
            const y = padT + plotH - ((val - minVal) / range) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        ctx.fillStyle = "#38bdf8";
        ctx.font = "11px JetBrains Mono";
        ctx.fillText(`Measured Radial Profile (Decay Slope: ${slope})`, padL + 10, padT + 20);

        ctx.fillStyle = "#94a3b8";
        ctx.fillText("Radial Spatial Frequency (r)", w / 2 - 50, h - 10);
    }

    // -------------------------------------------------------------
    // 10. Batch Pipeline Ingestion
    // -------------------------------------------------------------
    const batchInput = document.getElementById("batch-input");
    const batchTbody = document.getElementById("batch-tbody");
    const batchTableContainer = document.getElementById("batch-results-table-container");

    if (batchInput) {
        batchInput.addEventListener("change", async (e) => {
            if (!e.target.files || e.target.files.length === 0) return;
            if (batchTableContainer) batchTableContainer.style.display = "block";
            if (batchTbody) batchTbody.innerHTML = "";

            const files = Array.from(e.target.files);
            for (const file of files) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td><strong>${file.name}</strong></td>
                    <td>Scanning...</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td class="mono">-</td>
                `;
                batchTbody.appendChild(row);

                const fd = new FormData();
                fd.append("file", file);
                try {
                    const res = await fetch("/api/analyze/image", { method: "POST", body: fd });
                    if (res.ok) {
                        const d = await res.json();
                        const color = d.severity === "CRITICAL" ? "#ef4444" : (d.severity === "CLEAN" ? "#10b981" : "#f59e0b");
                        row.innerHTML = `
                            <td><strong>${d.media_name}</strong></td>
                            <td style="color:${color}; font-weight:700;">${d.verdict}</td>
                            <td><span style="color:${color};">${d.severity}</span></td>
                            <td class="mono">${(d.fused_fake_prob * 100).toFixed(1)}%</td>
                            <td class="mono">${d.confidence.toFixed(1)}%</td>
                            <td class="mono" style="font-size:0.75rem; color:#38bdf8;">${d.sha256 ? d.sha256.substring(0, 16) + '...' : '-'}</td>
                        `;
                    }
                } catch (err) {
                    row.innerHTML = `<td>${file.name}</td><td style="color:#ef4444;">Error</td><td>-</td><td>-</td><td>-</td><td>-</td>`;
                }
            }
        });
    }

    // -------------------------------------------------------------
    // Helper Utilities
    // -------------------------------------------------------------
    function downloadBlob(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    function jsonStringifySafe(obj) {
        return JSON.stringify(obj, (key, value) => {
            if (typeof value === "bigint") return value.toString();
            return value;
        });
    }
});
