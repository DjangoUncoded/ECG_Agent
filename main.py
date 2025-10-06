import base64
import traceback
import pandas as pd
import numpy as np
import io
import os
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import HTMLResponse
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for servers
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Load environment variables (for API key)
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load API key
API_KEY = os.getenv("GEMINI_API")
if not API_KEY:
    raise ValueError("GEMINI_API key not found. Please set it in your .env file.")

# Set up model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

# --- App Setup ---
UPLOAD_DIR = "uploads"
templates = Jinja2Templates(directory="templates")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = FastAPI()

# --- ECG Analysis Parameters ---

LOWCUT, HIGCUT, ORDER = 0.5, 40.0, 3


# --- Processing Functions ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


def analyze_ecg_file(file_path, fs=None):
    # Try reading CSV with headers first, fallback to no headers
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, header=None)

    # Pick first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV.")

    signal = df[numeric_cols[0]].dropna().values
    n_samples = len(signal)

    # Determine FS
    if fs is None:
        fs = 360  # default fallback

    time = np.arange(n_samples) / fs

    # Filter ECG
    filtered = bandpass_filter(signal, LOWCUT, HIGCUT, fs, ORDER)

    # Peak detection
    diffed = np.ediff1d(filtered, to_begin=0)
    squared = diffed ** 2
    ma_window = int(0.150 * fs)
    ma = np.convolve(squared, np.ones(ma_window) / ma_window, mode="same")
    threshold = np.mean(ma) + 0.5 * np.std(ma)
    distance = int(0.25 * fs)
    peaks, _ = find_peaks(ma, height=threshold, distance=distance)

    search_radius = int(0.05 * fs)
    refined_peaks = []
    for p in peaks:
        start, end = max(0, p - search_radius), min(len(filtered), p + search_radius)
        local_max = np.argmax(filtered[start:end]) + start
        refined_peaks.append(local_max)
    refined_peaks = np.array(sorted(set(refined_peaks)))

    peak_times = refined_peaks / fs
    rr_intervals = np.diff(peak_times)

    mean_hr = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    sdnn = np.std(rr_intervals, ddof=1) * 1000 if len(rr_intervals) > 1 else 0
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) * 1000 if len(rr_intervals) > 2 else 0

    # Plot
    buf = io.BytesIO()
    plt.figure(figsize=(12, 5))
    plt.plot(time, filtered, label="Filtered ECG", color="#007BFF")
    plt.plot(peak_times, filtered[refined_peaks], "rx", label="R-Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"ECG Analysis | Mean HR: {mean_hr:.2f} bpm")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "beats": len(refined_peaks),
        "mean_hr": f"{mean_hr:.2f}",
        "sdnn": f"{sdnn:.2f}",
        "rmssd": f"{rmssd:.2f}",
        "plot_url": f"data:image/png;base64,{img_base64}"
    }


# --- API Endpoints ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"❌ An error occurred: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=JSONResponse)
async def analyze_ecg(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        result = analyze_ecg_file(file_path)

        # ---------- MODIFIED PROMPT START ----------
        prompt_text = f"""
        Analyze the attached ECG plot and the following metrics to provide a concise summary.
        **The summary must be very brief.**

        **Format your response using simple Markdown.** Use '##' for main headings and '**' for bolding key terms.
        Start with a '## Summary' section, then a '## Heart Rate Variability (HRV) Metrics' section. Under the HRV section, briefly explain each metric,also in easy terms which is understandable to all audience and provide a suggestion if needed,pretent you are a doctor.

        - **Detected Beats:** {result['beats']}
        - **Mean Heart Rate (HR):** {result['mean_hr']} bpm
        - **SDNN:** {result['sdnn']} ms
        - **RMSSD:** {result['rmssd']} ms
        """
        # ---------- MODIFIED PROMPT END ----------

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": result["plot_url"]
                },
            ]
        )

        response = model.invoke([message])
        llm_summary = response.content

    finally:
        os.remove(file_path)

    result["summary"] = llm_summary
    return JSONResponse(content=result)