# Digit Classifier

A small feedforward neural network (784 → 128 → 64 → 10, ReLU) trained on MNIST with PyTorch, plus a drag-and-drop web page where you drop in images of digits and the network guesses them. Test accuracy is about 97.6%.

The web page runs the network **entirely in your browser** using the exported weights, so it can be hosted as a static site (on Vercel, for example).

## Project layout

| File | What it does |
| --- | --- |
| `digit_classifier.py` | Trains the network (10 epochs, batch size 64, Adam, cross-entropy), saves `model.pt`, then reloads it and prints sample predictions |
| `model.pt` | The trained weights |
| `export_weights.py` | Exports `model.pt` to `web/weights.json` for the browser |
| `web/index.html` | The drag-and-drop page (image preparation and network written in JavaScript) |
| `app.py` | Local server for the page, plus a `POST /predict` endpoint that runs the PyTorch model |
| `samples/` | Some MNIST test digits saved as PNGs to try in the page |

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python digit_classifier.py   # train (optional, model.pt is included)
.venv/bin/python export_weights.py     # refresh web/weights.json after retraining
.venv/bin/python app.py                # then open http://localhost:7860
```

## Deploy

`vercel.json` serves the `web/` folder as a static site with no build step. Only `web/` is uploaded (see `.vercelignore`).
