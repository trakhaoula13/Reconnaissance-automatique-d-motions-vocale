"use strict";

// true : l'analyse démarre dès que l'audio est ajouté ; false : il faut cliquer sur « Analyser ».
const AUTO_ANALYZE = false;

// Page servie par Flask (port 5000) : adresse relative. Sinon (Live Server, fichier ouvert...) : serveur local.
const API = location.port === "5000" ? "/api/analyze" : "http://localhost:5000/api/analyze";

const $ = (id) => document.getElementById(id);
const el = {
    picker: $("picker"),
    drop: $("dropzone"),
    file: $("file"),
    rec: $("rec"),
    recLabel: $("rec-label"),
    recTime: $("rec-time"),
    player: $("player"),
    audio: $("audio"),
    name: $("file-name"),
    clear: $("clear"),
    play: $("play"),
    back: $("back"),
    fwd: $("fwd"),
    cur: $("cur"),
    dur: $("dur"),
    pwave: $("pwave"),
    download: $("download"),
    mute: $("mute"),
    speed: $("speed"),
    restart: $("restart"),
    addFile: $("add-file"),
    addMic: $("add-mic"),
    addMicTime: $("add-mic-time"),
    analyze: $("analyze"),
    error: $("error"),
    result: $("result"),
    signal: $("signal"),
    fig: $("fig"),
};

const missing = Object.entries(el).filter(([, v]) => !v).map(([k]) => k);
if (missing.length) {
    document.body.insertAdjacentHTML("afterbegin", '<p style="margin:0;padding:12px 20px;background:#7f1d1d;color:#fff">' +
        `Fichiers de l'interface incohérents (éléments manquants : ${missing.join(", ")}). ` +
        "Remplacez index.html, style.css et script.js ensemble, puis rechargez avec Ctrl+F5.</p>");
    throw new Error("Éléments manquants : " + missing.join(", "));
}

if (location.port !== "5000") { // page ouverte via Live Server / fichier : risque de rechargement automatique
    document.body.insertAdjacentHTML("afterbegin", '<p style="margin:0;padding:8px 20px;background:#78350f;color:#fef3c7;font-size:.9rem">' +
        "Page ouverte hors du serveur Python. Pour éviter qu'elle se recharge toute seule après chaque analyse, " +
        "ouvrez <b>http://localhost:5000</b>.</p>");
}

// Toute erreur JavaScript devient visible sous le bouton « Analyser » (au lieu d'un échec silencieux)
window.addEventListener("error", (e) => showError("Erreur JavaScript : " + e.message));
window.addEventListener("unhandledrejection", (e) => showError("Erreur : " + ((e.reason && e.reason.message) || e.reason)));

// La page vient-elle d'être rechargée toute seule pendant qu'un audio était chargé ? (ex. Live Server)
try {
    const nav = performance.getEntriesByType("navigation")[0];
    if (nav && nav.type === "reload" && sessionStorage.getItem("ser-audio-loaded")) {
        document.body.insertAdjacentHTML("afterbegin", '<p style="margin:0;padding:8px 20px;background:#7f1d1d;color:#fff;font-size:.9rem">' +
            "La page a été rechargée automatiquement (extension Live Server ?). Ouvrez <b>http://localhost:5000</b> au lieu de Live Server.</p>");
    }
    sessionStorage.removeItem("ser-audio-loaded");
} catch { /* stockage indisponible : ignoré */ }

let wavBlob = null; // audio converti en WAV mono, envoyé au serveur
let peaks = null; // enveloppe pour les formes d'onde
let audioDur = 0; // durée de l'audio chargé (s)
let melData = null; // spectrogramme calculé, dessiné quand le résultat apparaît
let signalShown = false; // le panneau « analyse du signal » contient-il un dessin ?
let lastMel = null; // spectrogramme affiché (pour le redessiner au redimensionnement)
let runId = 0; // numéro de l'analyse en cours (ignore les réponses périmées)

/* ---------- Icônes (mêmes tracés que l'ancienne interface) ---------- */
const EYES = '<line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>';
const FACE = '<circle cx="12" cy="12" r="10"/>';
const ICONS = {
    neutral: FACE + EYES + '<line x1="8" x2="16" y1="15" y2="15"/>',
    happy: FACE + EYES + '<path d="M8 14s1.5 2 4 2 4-2 4-2"/>',
    sad: FACE + EYES + '<path d="M16 16s-1.5-2-4-2-4 2-4 2"/>',
    angry: FACE + '<path d="M16 16s-1.5-2-4-2-4 2-4 2"/><path d="M7.5 8 10 9"/><path d="m14 9 2.5-1"/><path d="M9 10h.01"/><path d="M15 10h.01"/>',
    fearful: FACE + EYES + '<path d="M7 7.5 10 6.5"/><path d="M17 7.5 14 6.5"/><ellipse cx="12" cy="16" rx="2" ry="2.5"/>',
    disgust: FACE + EYES + '<path d="M8 15.5c1-1.2 2 1.2 4 0s3 1.2 4 0"/>',
    surprised: FACE + EYES + '<circle cx="12" cy="15.5" r="2"/>',
    alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
};
const icon = (k) => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${ICONS[k]}</svg>`;

/* ---------- Choix du fichier ---------- */
el.drop.addEventListener("click", () => el.file.click());
el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault();
        el.file.click(); } });
el.file.addEventListener("change", () => el.file.files[0] && loadBlob(el.file.files[0], el.file.files[0].name));
["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault();
    el.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault();
    el.drop.classList.remove("over"); }));
el.drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadBlob(f, f.name); });
el.clear.addEventListener("click", reset);

/* ---------- Enregistrement micro ---------- */
let recorder = null,
    chunks = [],
    timer = null,
    t0 = 0;

el.rec.addEventListener("click", async() => {
    if (recorder) { recorder.stop(); return; }
    try {
        el.audio.pause();
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        chunks = [];
        recorder.ondataavailable = (e) => chunks.push(e.data);
        recorder.onstop = () => {
            stream.getTracks().forEach((t) => t.stop());
            clearInterval(timer);
            el.rec.classList.remove("recording");
            el.addMic.classList.remove("recording");
            el.recLabel.textContent = "Enregistrer ma voix";
            el.recTime.textContent = el.addMicTime.textContent = "";
            const blob = new Blob(chunks, { type: recorder.mimeType });
            recorder = null;
            loadBlob(blob, "enregistrement.wav");
        };
        recorder.start();
        t0 = Date.now();
        el.rec.classList.add("recording");
        el.addMic.classList.add("recording");
        el.recLabel.textContent = "Arrêter";
        timer = setInterval(() => {
            const s = Math.floor((Date.now() - t0) / 1000);
            el.recTime.textContent = el.addMicTime.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
        }, 250);
    } catch {
        showError("Micro inaccessible. Autorisez le micro dans le navigateur (la page doit être ouverte en localhost ou en HTTPS).");
    }
});

/* ---------- Décodage + conversion en WAV mono 16 bits ---------- */
async function loadBlob(blob, name) {
    hideError();
    try {
        const ctx = new(window.AudioContext || window.webkitAudioContext)();
        const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
        ctx.close();
        const mono = new Float32Array(buf.length);
        for (let c = 0; c < buf.numberOfChannels; c++) {
            const d = buf.getChannelData(c);
            for (let i = 0; i < d.length; i++) mono[i] += d[i] / buf.numberOfChannels;
        }
        wavBlob = encodeWav(mono, buf.sampleRate);
        peaks = computePeaks(mono, 800);
        audioDur = buf.duration;
        el.audio.pause();
        el.audio.src = URL.createObjectURL(wavBlob);
        el.name.textContent = name;
        resetControls();
        try { sessionStorage.setItem("ser-audio-loaded", "1"); } catch { /* ignoré */ }
        melData = computeMel(mono, buf.sampleRate);
        el.picker.hidden = true; // la forme d'onde prend la place de la zone d'ajout
        el.player.hidden = false;
        el.analyze.disabled = false;
        updatePlayer();
    } catch {
        reset();
        showError("Ce fichier audio n'a pas pu être lu. Essayez un fichier WAV ou MP3.");
        return;
    }
    if (AUTO_ANALYZE) runAnalysis();
}

function encodeWav(samples, sr) {
    const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
    const str = (o, s) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
    str(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    str(8, "WAVEfmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    str(36, "data");
    view.setUint32(40, samples.length * 2, true);
    samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
    return new Blob([view], { type: "audio/wav" });
}

function computePeaks(data, n) {
    const step = Math.max(1, Math.floor(data.length / n)),
        out = [];
    for (let i = 0; i < n; i++) {
        let mx = 0;
        for (let j = i * step; j < Math.min(data.length, (i + 1) * step); j++) mx = Math.max(mx, Math.abs(data[j]));
        out.push(mx);
    }
    const top = Math.max(...out, 1e-9);
    return out.map((v) => v / top);
}

/* ---------- Analyse (appel au serveur Python) ---------- */
async function runAnalysis() {
    if (!wavBlob) return;
    const id = ++runId;
    hideError();
    el.analyze.disabled = true;
    el.analyze.textContent = "Analyse en cours…";
    showPlaceholder(); // résultat et analyse du signal restent vides
    try { clearSignal(); } catch (err) { console.error(err); }
    let reached = false; // le serveur a-t-il répondu ?
    const form = new FormData();
    form.append("audio", wavBlob, "audio.wav");
    form.append("model", document.querySelector('input[name="model"]:checked').value);
    try {
        const res = await fetch(API, { method: "POST", body: form });
        reached = true;
        const raw = await res.text();
        let data = null;
        try { data = JSON.parse(raw); } catch { /* réponse vide ou non JSON */ }
        if (id !== runId) return;
        if (!data) throw new Error(`Réponse invalide du serveur (code ${res.status}). Lancez « python backend/server.py » et regardez le terminal.`);
        if (!res.ok) throw new Error(data.error || "Erreur du serveur.");
        showResult(data);
        try { signalShown = true;
            drawFigure(melData); } catch (err) { signalShown = false;
            console.error("Figure du signal :", err); } // le résultat reste affiché
    } catch (e) {
        if (id !== runId) return;
        showPlaceholder();
        showError(e instanceof TypeError && !reached ? "Serveur injoignable. Lancez « python backend/server.py » puis ouvrez http://localhost:5000." : e.message);
    } finally {
        if (id === runId) { el.analyze.disabled = !wavBlob;
            el.analyze.textContent = "Analyser"; }
    }
}
el.analyze.addEventListener("click", runAnalysis);
// L'analyse ne démarre que par un clic sur « Analyser » (changer d'approche ne la relance pas).

function showResult({ probs, labels, model }) {
    const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const [top, pTop] = ranked[0];
    const warn = pTop < 0.4 ?
        `<div class="warn">${icon("alert")}<span>Confiance faible : émotion ambiguë ou audio de mauvaise qualité.</span></div>` : "";
    const rows = ranked.map(([k, p]) => `
    <div class="row ${k === top ? "best" : ""}">${icon(k)}
      <span class="lab">${labels[k]}</span>
      <div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div>
      <span class="pct">${Math.round(p * 100)}%</span>
    </div>`).join("");
    el.result.className = "card";
    el.result.hidden = false;
    el.result.innerHTML = `
    <div class="top">${icon(top)}
      <div><small>Émotion détectée</small><div class="name">${labels[top]}</div>
      <small>Confiance : ${Math.round(pTop * 100)}%</small></div></div>
    ${warn}<hr>${rows}<div class="foot">Modèle : ${model}</div>`;
}

/* ---------- Spectrogramme Mel calculé dans le navigateur ---------- */
function fft(re, im) { // FFT radix-2 en place
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len,
            wr = Math.cos(ang),
            wi = Math.sin(ang),
            h = len >> 1;
        for (let i = 0; i < n; i += len) {
            let cr = 1,
                ci = 0;
            for (let k = 0; k < h; k++) {
                const a = i + k,
                    b = a + h;
                const vr = re[b] * cr - im[b] * ci,
                    vi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - vr;
                im[b] = im[a] - vi;
                re[a] += vr;
                im[a] += vi;
                const t = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = t;
            }
        }
    }
}

function computeMel(x, sr, nMels = 64, nFft = 1024, hop = 512) {
    const fmax = Math.min(8192, sr / 2);
    const toMel = (f) => 2595 * Math.log10(1 + f / 700),
        toHz = (m) => 700 * (10 ** (m / 2595) - 1);
    const bins = Array.from({ length: nMels + 2 }, (_, i) =>
        Math.floor((nFft + 1) * toHz((toMel(fmax) * i) / (nMels + 1)) / sr));
    const win = Float32Array.from({ length: nFft }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft));
    const n = Math.min(x.length, Math.floor(8 * sr)); // 8 s maximum, comme le serveur
    const frames = Math.max(1, Math.floor((n - nFft) / hop) + 1);
    const re = new Float32Array(nFft),
        im = new Float32Array(nFft),
        pw = new Float32Array(nFft / 2 + 1);
    const db = Array.from({ length: nMels }, () => new Float32Array(frames));
    let max = -Infinity;
    for (let t = 0; t < frames; t++) {
        for (let i = 0; i < nFft; i++) { re[i] = (x[t * hop + i] || 0) * win[i];
            im[i] = 0; }
        fft(re, im);
        for (let k = 0; k < pw.length; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
        for (let m = 0; m < nMels; m++) {
            const lo = bins[m],
                mid = bins[m + 1],
                hi = bins[m + 2];
            let s = 0;
            if (hi <= lo + 1) s = pw[mid];
            else
                for (let k = lo; k < hi; k++) s += pw[k] * (k < mid ? (k - lo) / Math.max(1, mid - lo) : (hi - k) / Math.max(1, hi - mid));
            db[m][t] = 10 * Math.log10(s + 1e-10);
            if (db[m][t] > max) max = db[m][t];
        }
    }
    const out = db.map((row) => Array.from(row, (v) => Math.round(255 * Math.max(0, (v - (max - 80)) / 80)))); // 80 dB de dynamique
    out.fmax = fmax;
    out.dur = n / sr;
    return out;
}

/* ---------- Lecteur audio avec forme d'onde ---------- */
const PLAY = '<polygon points="6 3 20 12 6 21 6 3"/>';
const PAUSE = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
const fmt = (s) => { s = Math.max(0, Math.floor(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

function updatePlayer() {
    const d = el.audio.duration || 0,
        t = el.audio.currentTime || 0;
    el.cur.textContent = fmt(t);
    el.dur.textContent = fmt(d);
    el.play.firstElementChild.innerHTML = el.audio.paused ? PLAY : PAUSE;
    if (!peaks) return;
    const g = fitCanvas(el.pwave),
        w = el.pwave.clientWidth,
        h = Number(el.pwave.dataset.h);
    const n = Math.max(20, Math.floor(w / 6)),
        prog = d ? t / d : 0;
    for (let i = 0; i < n; i++) {
        const a = Math.floor((i / n) * peaks.length),
            b = Math.max(a + 1, Math.floor(((i + 1) / n) * peaks.length));
        const p = Math.max(...peaks.slice(a, b)),
            bh = Math.max(3, p * h * 0.95);
        g.fillStyle = i / n < prog ? "#818cf8" : "#64748b";
        g.fillRect((i / n) * w, (h - bh) / 2, Math.max(2, (w / n) * 0.55), bh);
    }
    g.fillStyle = "#e2e8f0";
    g.fillRect(Math.min(w - 2, prog * w), 0, 2, h); // curseur de lecture
}

function tick() { updatePlayer(); if (!el.audio.paused) requestAnimationFrame(tick); }
el.audio.addEventListener("play", tick);
["pause", "ended", "loadedmetadata", "seeked"].forEach((ev) => el.audio.addEventListener(ev, updatePlayer));
el.play.addEventListener("click", () => (el.audio.paused ? el.audio.play() : el.audio.pause()));
el.back.addEventListener("click", () => { el.audio.currentTime = Math.max(0, el.audio.currentTime - 2); });
el.fwd.addEventListener("click", () => { el.audio.currentTime = Math.min(el.audio.duration || 0, el.audio.currentTime + 2); });
const MUTE = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="22" x2="16" y1="9" y2="15"/><line x1="16" x2="22" y1="9" y2="15"/>';
const VOL = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>';
const RATES = [1, 1.5, 2, 0.5];

function resetControls() {
    el.audio.muted = false;
    el.audio.playbackRate = 1;
    el.mute.firstElementChild.innerHTML = VOL;
    el.speed.textContent = "1x";
}
el.mute.addEventListener("click", () => {
    el.audio.muted = !el.audio.muted;
    el.mute.firstElementChild.innerHTML = el.audio.muted ? MUTE : VOL;
});
el.speed.addEventListener("click", () => {
    const r = RATES[(RATES.indexOf(el.audio.playbackRate) + 1) % RATES.length];
    el.audio.playbackRate = r;
    el.speed.textContent = `${r}x`;
});
el.restart.addEventListener("click", () => { el.audio.currentTime = 0;
    updatePlayer(); });
el.download.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = el.audio.src;
    a.download = el.name.textContent.replace(/\.[^.]+$/, "") + ".wav";
    a.click();
});
el.addFile.addEventListener("click", () => el.file.click());
el.addMic.addEventListener("click", () => el.rec.click()); // même enregistrement que le bouton de départ
el.pwave.addEventListener("click", (e) => {
    const r = el.pwave.getBoundingClientRect();
    if (el.audio.duration) el.audio.currentTime = ((e.clientX - r.left) / r.width) * el.audio.duration;
});

/* ---------- Dessins (canvas) ---------- */
function fitCanvas(c) {
    const dpr = window.devicePixelRatio || 1;
    const h = Number(c.dataset.h) || (c.dataset.h = parseInt(c.getAttribute("height"), 10)); // hauteur d'origine, mémorisée une fois
    c.style.height = h + "px";
    c.width = Math.round(c.clientWidth * dpr);
    c.height = Math.round(h * dpr);
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return g;
}



/* ---------- Figure « Forme d'onde + Spectrogramme Mel » (même présentation que l'ancienne interface) ---------- */
const COOLWARM = [
    [59, 76, 192],
    [141, 176, 254],
    [221, 221, 221],
    [244, 154, 123],
    [180, 4, 38]
];

function coolwarm(v) {
    const t = (v / 255) * (COOLWARM.length - 1),
        i = Math.min(COOLWARM.length - 2, Math.floor(t)),
        f = t - i;
    return COOLWARM[i].map((s, k) => Math.round(s + (COOLWARM[i + 1][k] - s) * f));
}

function drawFigure(mel) { // mel : [bandes][temps] de valeurs 0-255 (bande 0 = grave), + mel.fmax, mel.dur
    lastMel = mel;
    const g = fitCanvas(el.fig),
        W = el.fig.clientWidth,
        H = Number(el.fig.dataset.h);
    const L = 64,
        R = 18,
        FONT = '"DejaVu Sans", Verdana, Arial, sans-serif';
    const setFont = (px) => { g.font = `${px}px ${FONT}`; };
    g.fillStyle = "#fff";
    g.fillRect(0, 0, W, H);
    g.strokeStyle = "#000";
    g.lineWidth = 0.8;

    const frame = (top, height, title) => { // cadre noir + titre
        g.fillStyle = "#000";
        g.textAlign = "center";
        g.textBaseline = "alphabetic";
        setFont(14);
        g.fillText(title, (L + W - R) / 2, top - 10);
        g.strokeRect(L + 0.5, top + 0.5, W - R - L, height);
        return { x0: L, x1: W - R, y0: top, y1: top + height };
    };
    const timeAxis = (b, dur) => { // graduations en secondes + « Time »
        const step = [0.1, 0.25, 0.5, 1, 2, 5].find((s) => dur / s <= 5) || 5;
        g.fillStyle = "#000";
        g.textAlign = "center";
        g.textBaseline = "alphabetic";
        setFont(11);
        for (let t = 0; t <= dur + 1e-9; t += step) {
            const x = b.x0 + (t / dur) * (b.x1 - b.x0);
            g.beginPath();
            g.moveTo(x + 0.5, b.y1 + 0.5);
            g.lineTo(x + 0.5, b.y1 + 5);
            g.stroke();
            g.fillText(String(+t.toFixed(2)), x, b.y1 + 19);
        }
        setFont(12);
        g.fillText("Time", (b.x0 + b.x1) / 2, b.y1 + 36);
    };
    const yTick = (b, y, label) => {
        g.beginPath();
        g.moveTo(b.x0 - 5, y + 0.5);
        g.lineTo(b.x0, y + 0.5);
        g.stroke();
        g.fillStyle = "#000";
        g.textAlign = "right";
        g.textBaseline = "middle";
        setFont(11);
        g.fillText(label, b.x0 - 8, y);
    };

    // 1) Forme d'onde
    const h1 = 150,
        b1 = frame(36, h1, "Forme d'onde"),
        yc = b1.y0 + h1 / 2,
        half = h1 / 2 / 1.05;
    const wd = audioDur || mel.dur,
        n = peaks.length,
        cw = (b1.x1 - b1.x0) / n;
    g.fillStyle = "#2f6fb0";
    peaks.forEach((p, i) => g.fillRect(b1.x0 + i * cw, yc - p * half, cw + 0.4, Math.max(1, 2 * p * half)));
    [1, 0.5, 0, -0.5, -1].forEach((v) => yTick(b1, yc - v * half, (v < 0 ? "−" : "") + Math.abs(v).toFixed(1)));
    timeAxis(b1, wd);

    // 2) Spectrogramme Mel (axe des fréquences en échelle Mel : 0, 512, 1024, 2048, 4096, 8192 Hz)
    const h2 = 190,
        b2 = frame(262, h2, "Spectrogramme Mel");
    const rows = mel.length,
        cols = mel[0].length,
        mw = (b2.x1 - b2.x0) / cols,
        mh = h2 / rows;
    for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
            g.fillStyle = `rgb(${coolwarm(mel[r][c]).join(",")})`;
            g.fillRect(b2.x0 + c * mw, b2.y1 - (r + 1) * mh, mw + 0.6, mh + 0.6);
        }
    g.strokeStyle = "#000";
    g.strokeRect(L + 0.5, b2.y0 + 0.5, W - R - L, h2);
    const toMel = (f) => 2595 * Math.log10(1 + f / 700);
    [0, 512, 1024, 2048, 4096, 8192].filter((f) => f <= mel.fmax + 1)
        .forEach((f) => yTick(b2, b2.y1 - (toMel(f) / toMel(mel.fmax)) * h2, String(f)));
    g.save();
    g.translate(14, (b2.y0 + b2.y1) / 2);
    g.rotate(-Math.PI / 2);
    g.fillStyle = "#000";
    g.textAlign = "center";
    g.textBaseline = "alphabetic";
    setFont(12);
    g.fillText("Hz", 0, 0);
    g.restore();
    timeAxis(b2, mel.dur);
}

function clearSignal() { signalShown = false;
    lastMel = null;
    fitCanvas(el.fig); }

window.addEventListener("resize", () => {
    if (peaks && !el.player.hidden) updatePlayer();
    if (signalShown && lastMel) drawFigure(lastMel);
    else clearSignal();
});

/* ---------- Utilitaires ---------- */
function showError(msg) { el.error.textContent = msg;
    el.error.hidden = false; }

function hideError() { el.error.hidden = true; }

function showPlaceholder() { el.result.className = "card";
    el.result.textContent = ""; }

function reset() {
    runId++; // ignore une analyse en cours
    el.audio.pause();
    wavBlob = null;
    peaks = null;
    audioDur = 0;
    el.file.value = "";
    el.audio.removeAttribute("src");
    melData = null;
    el.picker.hidden = false;
    el.player.hidden = true;
    el.analyze.disabled = true;
    hideError();
    showPlaceholder();
    clearSignal();
}
reset();