const API_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
    loadModelInfo();
    initRadioButtons();
    initToggleButtons();
    initSosmedChips();
    initRangeSliders();
    initForm();
    initCloseResult();
});

async function loadModelInfo() {
    try {
        const res = await fetch(`${API_URL}/model-info`);
        const data = await res.json();
        document.getElementById("modelName").textContent =
            `${data.model_name} • Akurasi ${(data.test_accuracy * 100).toFixed(1)}%`;
    } catch {
        document.getElementById("modelName").textContent = "API Offline";
        document.querySelector(".badge-dot").style.background = "#f87171";
    }
}

function initRadioButtons() {
    const radios = document.querySelectorAll('.radio-btn input[type="radio"]');
    radios.forEach(radio => {
        radio.addEventListener("change", () => {
            document.querySelectorAll(".radio-btn").forEach(btn => btn.classList.remove("active"));
            radio.closest(".radio-btn").classList.add("active");
        });
    });
}

function initToggleButtons() {
    const toggles = [
        { id: "cemas", labelId: "cemasLabel" },
        { id: "noPurpose", labelId: "noPurposeLabel" },
        { id: "disruptive", labelId: "disruptiveLabel" },
        { id: "tryReduce", labelId: "tryReduceLabel" },
    ];
    toggles.forEach(({ id, labelId }) => {
        const input = document.getElementById(id);
        const label = document.getElementById(labelId);
        const update = () => { label.textContent = input.checked ? "Iya" : "Tidak"; };
        update();
        input.addEventListener("change", update);
    });
}

function initSosmedChips() {
    document.querySelectorAll(".sosmed-chip input").forEach(input => {
        input.addEventListener("change", () => {
            input.closest(".sosmed-chip").classList.toggle("active", input.checked);
        });
    });
}

function initRangeSliders() {
    const sleepTime = document.getElementById("sleepTime");
    const sleepVal = document.getElementById("sleepTimeVal");
    const focusHours = document.getElementById("focusHours");
    const focusVal = document.getElementById("focusHoursVal");

    const updateSleep = () => {
        const v = parseFloat(sleepTime.value);
        const h = Math.floor(v);
        const m = (v % 1) * 60;
        sleepVal.textContent = `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0").slice(0, 2)}`;
    };

    const updateFocus = () => {
        focusVal.textContent = `${parseFloat(focusHours.value).toFixed(1)} jam`;
    };

    sleepTime.addEventListener("input", updateSleep);
    focusHours.addEventListener("input", updateFocus);
    updateSleep();
    updateFocus();
}

function initForm() {
    document.getElementById("predictionForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("predictBtn");
        btn.classList.add("loading");
        btn.disabled = true;

        const status = document.querySelector('input[name="status"]:checked').value;
        const payload = {
            Usia: parseFloat(document.getElementById("usia").value),
            Rata_rata_screen_time_perhari: parseFloat(document.getElementById("screenTime").value),
            Frekuensi_membuka_HP_perhari: parseFloat(document.getElementById("hpFreq").value),
            Jumlah_aplikasi_media_sosial_aktif_yang_digunakan: parseFloat(document.getElementById("appCount").value),
            Durasi_waktu_tidur_perhari: parseFloat(document.getElementById("sleepDur").value),
            Jam_tidur_hours: parseFloat(document.getElementById("sleepTime").value),
            Tidur_larut_karena_sosial_media: parseInt(document.getElementById("sleepLate").value),
            Jam_waktu_belajar_bekerja_fokus_sehari: parseFloat(document.getElementById("focusHours").value),
            Sulit_fokus_tanpa_mengecek_HP: parseInt(document.getElementById("diffFocus").value),
            Sering_membuka_media_sosial_saat_belajar_bekerja: parseInt(document.getElementById("openSosmed").value),
            Merasa_cemas_jika_jauh_dari_HP: document.getElementById("cemas").checked ? 1 : 0,
            Sering_membuka_sosial_media_tanpa_tujuan_jelas: document.getElementById("noPurpose").checked ? 1 : 0,
            Penggunaan_media_sosial_mengganggu_belajar_kerja: document.getElementById("disruptive").checked ? 1 : 0,
            Pernah_mencoba_mengurangi_penggunaan_media_sosial_namun_gagal: document.getElementById("tryReduce").checked ? 1 : 0,
            Status_Mahasiswa: status === "Mahasiswa" ? 1 : 0,
            Status_Pelajar: status === "Pelajar" ? 1 : 0,
            Status_Pekerja: status === "Pekerja" ? 1 : 0,
            Sosmed_Instagram: document.getElementById("sosmedInstagram").checked ? 1 : 0,
            Sosmed_Tiktok: document.getElementById("sosmedTiktok").checked ? 1 : 0,
            Sosmed_Youtube: document.getElementById("sosmedYoutube").checked ? 1 : 0,
            Sosmed_X_twitter: document.getElementById("sosmedTwitter").checked ? 1 : 0,
            Sosmed_Facebook: document.getElementById("sosmedFacebook").checked ? 1 : 0,
            Sosmed_Threads: document.getElementById("sosmedThreads").checked ? 1 : 0,
            Sosmed_WhatsApp: document.getElementById("sosmedWhatsapp").checked ? 1 : 0,
        };

        try {
            const res = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            showResult(data);
        } catch (err) {
            alert("Gagal terhubung ke API. Pastikan FastAPI berjalan di " + API_URL);
        } finally {
            btn.classList.remove("loading");
            btn.disabled = false;
        }
    });
}

function showResult(data) {
    const section = document.getElementById("resultSection");
    section.style.display = "block";
    section.scrollIntoView({ behavior: "smooth", block: "center" });

    const emojis = { "Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴" };
    const colors = { "Rendah": "#34d399", "Sedang": "#fbbf24", "Tinggi": "#f87171" };
    const className = data.class_name;
    const proba = data.probabilities;
    const maxProba = Math.max(...Object.values(proba));

    document.getElementById("gaugeEmoji").textContent = emojis[className];
    document.getElementById("gaugeLabel").textContent = className;

    const fill = document.getElementById("gaugeFill");
    fill.style.stroke = colors[className];
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (maxProba * circumference);
    setTimeout(() => { fill.style.strokeDashoffset = offset; }, 100);

    const confEl = document.getElementById("confidenceValue");
    animateNumber(confEl, 0, Math.round(maxProba * 100), 1000, "%");

    const barsContainer = document.getElementById("probaBars");
    barsContainer.innerHTML = "";

    const classOrder = ["Rendah", "Sedang", "Tinggi"];
    classOrder.forEach((cls) => {
        const val = proba[cls] || 0;
        const item = document.createElement("div");
        item.className = "proba-item";
        item.innerHTML = `
            <div class="proba-label">
                <span>${emojis[cls]}</span>
                <span>${cls}</span>
            </div>
            <div class="proba-bar-bg">
                <div class="proba-bar-fill ${cls.toLowerCase()}" style="width: 0%"></div>
            </div>
            <div class="proba-value">${(val * 100).toFixed(1)}%</div>
        `;
        barsContainer.appendChild(item);
        setTimeout(() => {
            item.querySelector(".proba-bar-fill").style.width = `${val * 100}%`;
        }, 200);
    });
}

function animateNumber(el, start, end, duration, suffix = "") {
    const range = end - start;
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + range * eased);
        el.textContent = current + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function initCloseResult() {
    document.getElementById("closeResult").addEventListener("click", () => {
        const section = document.getElementById("resultSection");
        section.style.display = "none";
        const fill = document.getElementById("gaugeFill");
        fill.style.strokeDashoffset = 339.29;
    });
}
