"""Export model.pt's weights to web/weights.json so the browser can run the network.

Run after training:  .venv/bin/python export_weights.py
"""

import json
from pathlib import Path

import torch

from digit_classifier import DigitNet, MODEL_PATH

HERE = Path(__file__).parent

model = DigitNet()
model.load_state_dict(torch.load(HERE / MODEL_PATH, map_location="cpu", weights_only=True))

# The three Linear layers, in order (net.1, net.3, net.5 inside the Sequential)
layers = []
for module in model.net:
    if isinstance(module, torch.nn.Linear):
        layers.append({
            "in": module.in_features,
            "out": module.out_features,
            # Row-major [out][in], flattened; rounded to keep the file small (~1 MB)
            "weight": [round(v, 6) for v in module.weight.detach().flatten().tolist()],
            "bias": [round(v, 6) for v in module.bias.detach().tolist()],
        })

out = HERE / "web" / "weights.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"mean": 0.1307, "std": 0.3081, "layers": layers}, separators=(",", ":")))
print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
