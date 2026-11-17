"""Local web UI: drag a digit image onto the page and the trained network guesses it.

Run:  .venv/bin/python app.py   then open http://localhost:7860
Serves the static page in web/ (which runs the network in the browser) plus a
POST /predict endpoint that runs the PyTorch model on the server.
"""

import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from digit_classifier import DigitNet, MODEL_PATH

PORT = 7860
HERE = Path(__file__).parent

# Load the trained network once at startup (CPU is plenty fast for single images)
model = DigitNet()
model.load_state_dict(torch.load(HERE / MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()


def to_mnist(img: Image.Image) -> np.ndarray:
    """Turn an arbitrary photo/drawing of a digit into a 28x28 MNIST-style array (0-1)."""
    gray = ImageOps.exif_transpose(img).convert("L")

    # Flatten transparency onto white so PNGs with clear backgrounds work
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("L", img.size, 255)
        bg.paste(gray, mask=img.getchannel("A"))
        gray = bg

    arr = np.asarray(gray, dtype=np.float32) / 255.0

    # MNIST is white digit on black; invert if the background looks light
    if np.median(arr) > 0.5:
        arr = 1.0 - arr

    # Stretch contrast and drop faint background noise
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
    arr[arr < 0.2] = 0.0

    # Crop to the digit's bounding box
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Fit into a 20x20 box keeping aspect ratio (as MNIST does)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = Image.fromarray((arr * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    # Paste onto 28x28 canvas, then shift so the centre of mass is in the middle
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(digit, dtype=np.float32) / 255.0
    total = canvas.sum()
    if total > 0:
        cy = (canvas.sum(1) * np.arange(28)).sum() / total
        cx = (canvas.sum(0) * np.arange(28)).sum() / total
        canvas = np.roll(canvas, (round(14 - cy), round(14 - cx)), axis=(0, 1))
    return canvas


def predict(image_bytes: bytes) -> dict:
    """Run the network on raw image bytes; return probabilities and the processed preview."""
    arr = to_mnist(Image.open(io.BytesIO(image_bytes)))
    x = (torch.from_numpy(arr) - 0.1307) / 0.3081          # same normalisation as training
    with torch.no_grad():
        probs = torch.softmax(model(x.view(1, 1, 28, 28)), dim=1)[0].tolist()

    # Send back what the network actually "saw" so you can sanity-check preprocessing
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).resize((140, 140), Image.NEAREST).save(buf, "PNG")
    return {
        "digit": int(np.argmax(probs)),
        "probs": probs,
        "preview": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve the same static page Vercel serves (it runs the network in the browser)
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/index.html": ("index.html", "text/html; charset=utf-8"),
                 "/weights.json": ("weights.json", "application/json")}
        if self.path in files:
            name, ctype = files[self.path]
            self._send(200, (HERE / "web" / name).read_bytes(), ctype)
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, b"Not found", "text/plain")
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            result = predict(body)
            self._send(200, json.dumps(result).encode(), "application/json")
        except Exception as e:  # e.g. not an image
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    print(f"Digit classifier UI running at http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
