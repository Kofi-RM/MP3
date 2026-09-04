# Model Compare

A local Flask website for comparing YOLO object detection with Vision Transformer
image classification. YOLO Studio also supports webcam and video-file detection.

**No paid hosting, Render account, Docker, or API key is required.**

## Run on this computer

The dependencies are already installed on this computer.

1. Open `C:\Users\kofir\Flask website\MP3` in File Explorer.
2. Double-click **`start.cmd`**.
3. Wait until the terminal says `Running on http://127.0.0.1:5000`.
4. Open **http://127.0.0.1:5000** in your browser.

Keep the terminal window open while using the website. To stop the server, press
**Ctrl+C** in that window. If the site is already running, just open the URL;
you do not need to start a second copy.

Use `start.cmd`, not VS Code Live Server. Live Server cannot run the Python AI
models or handle their upload requests.

## First-time setup on another Windows computer

You need:

- Python **3.13** with the Windows Python launcher (`py`).
- A copy of this repository, including `yolov8n.pt`.
- Internet access for the initial package installation and model download.
- Enough free disk space for the AI libraries and model weights (allow several GB).

Open PowerShell **inside the MP3 repository folder**, then run:

```powershell
# Create the Python environment next to the MP3 folder.
# This matches the location used by start.cmd.
py -3.13 -m venv ..\.venv

# Install the application's dependencies.
& "..\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# Start the website.
.\start.cmd
```

If that parent `.venv` already exists and belongs to this app, reuse it rather
than recreating it. No environment activation is needed for these commands.

The first launch downloads the ViT model and can take longer. Later launches
reuse its local cache. The app runs on the CPU; a dedicated GPU is not required.

## Using the website

### Compare an image

Open http://127.0.0.1:5000/compare, choose a JPG, PNG, or GIF, then click
**Compare models**. Results include object detections, classifications, confidence
scores, and processing times.

Images must be no larger than 16 MB and 8 million pixels. Large accepted images
are resized for analysis and display.

### Use YOLO with your webcam

Open http://127.0.0.1:5000/yolo, select **Webcam**, and click **Start webcam**.
Allow camera permission when your browser asks. Click **Stop** to release the
camera. Audio is not requested.

### Analyze a video

In YOLO Studio, select **Video file**, choose a browser-supported video such as
MP4 (H.264) or WebM, and click **Analyze video**. The source player supports pause,
seek, and replay. Adjust the confidence slider to filter detections.

Detection uses sampled frames, so the analyzed view may update more slowly than
the original video. Switching away from the browser tab stops analysis; press
Start again when you return.

## Privacy and local access

The server binds to `127.0.0.1`, so it is accessible only from this computer by
default. Image and video frames are processed by your local Python server, not
sent to a hosted AI service. The app does not retain uploads or camera recordings;
multipart uploads may use temporary files during request handling.

Initial setup contacts package/model download services. The design also loads
fonts from Google Fonts; fallback fonts are used if they cannot load.

## Troubleshooting

- **The page cannot be reached:** start `start.cmd` and wait for the running-server
  message. Check the terminal for errors.
- **Missing module:** run the dependency installation command above using the
  environment's Python, not a different Python installation.
- **Python 3.13 not found:** install it, then retry. If `py` is unavailable, use
  the full path to your Python 3.13 executable in the environment-creation command.
- **The first launch seems slow:** the model may still be downloading or loading.
  Keep the terminal open and check its messages.
- **Camera denied or unavailable:** allow camera access in browser site settings,
  close other apps using it, and try Chrome or Edge directly.
- **Video will not play:** try an MP4 encoded with H.264 or a WebM video.
- **Models busy:** another analysis is running. Wait briefly and retry; the app
  processes one inference request at a time to limit memory use.
- **Port 5000 already in use:** an existing copy may already be running. Open its
  URL, or stop that copy before launching another.

## Editing the site

See `PAGE-GUIDE.md` for the HTML, CSS, JavaScript, and Python file layout. Restart
the server after changing Python code or HTML templates, then refresh the browser.

The optional Render/Docker files are not needed for local use. You can ignore them.
