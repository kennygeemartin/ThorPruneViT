"""Minimal ThorPruneViT web server with real PyTorch inference.

Set THORPRUNEVIT_CHECKPOINT to a directory containing checkpoint.pt (as produced
by save_pruned_model), then run: python app.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_SOURCE = ROOT / "Model Training" / "reproducibility_fixed" / "src"
sys.path.insert(0, str(MODEL_SOURCE))

from thorprunevit.model import count_parameters, load_pruned_model  # noqa: E402

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]
MAX_IMAGE_BYTES = 15 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.checkpoint_dir = os.environ.get("THORPRUNEVIT_CHECKPOINT", "").strip()
        self.error = None
        if not self.checkpoint_dir:
            self.error = "THORPRUNEVIT_CHECKPOINT is not configured."
            return
        try:
            self.model = load_pruned_model(self.checkpoint_dir, map_location=DEVICE).to(DEVICE).eval()
            if self.model.head.out_features != len(LABELS):
                raise ValueError(f"Checkpoint has {self.model.head.out_features} outputs; expected 14 NIH labels")
        except Exception as exc:  # surfaced through /api/status; never substitute fake output
            self.error = f"Checkpoint could not be loaded: {exc}"
            self.model = None

    def status(self) -> dict:
        if self.model is None:
            return {"ready": False, "device": str(DEVICE), "error": self.error, "labels": LABELS}
        structure = self.model.active_structure()
        return {
            "ready": True,
            "device": str(DEVICE),
            "model": "ThorPruneViT (structurally pruned ViT-B/16)",
            "parameters": count_parameters(self.model),
            "heads_per_layer": structure["heads_per_layer"],
            "ffn_dims": structure["ffn_dims"],
            "labels": LABELS,
        }

    def predict(self, raw: bytes) -> dict:
        if self.model is None:
            raise RuntimeError(self.error or "Model is unavailable")
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 15 MB limit")
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8).reshape(224, 224, 3)
        x = pixels.permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        x = ((x - mean) / std).requires_grad_(True)

        started = time.perf_counter()
        logits = self.model(x)
        probabilities = torch.sigmoid(logits)
        top_logit = logits[0, probabilities[0].argmax()]
        self.model.zero_grad(set_to_none=True)
        top_logit.backward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000

        saliency = x.grad.detach().abs().amax(dim=1, keepdim=True)
        saliency = F.adaptive_avg_pool2d(saliency, (14, 14))[0, 0]
        saliency -= saliency.min()
        saliency /= saliency.max().clamp_min(1e-8)
        return {
            "labels": LABELS,
            "probabilities": probabilities[0].detach().cpu().tolist(),
            "saliency": saliency.cpu().tolist(),
            "inference_time_ms": round(elapsed_ms, 2),
            "device": str(DEVICE),
        }


SERVICE = ModelService()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.path = "/ThorPruneViT-Clinical-Interface.html"
        if self.path == "/api/status":
            return self._json(SERVICE.status(), 200 if SERVICE.model else 503)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_IMAGE_BYTES * 2:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            encoded = body.get("image", "")
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            result = SERVICE.predict(base64.b64decode(encoded, validate=True))
            self._json(result)
        except (ValueError, KeyError, json.JSONDecodeError, UnidentifiedImageError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as exc:
            self._json({"error": f"Inference failed: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"ThorPruneViT server: http://127.0.0.1:{port}")
    print(json.dumps(SERVICE.status(), indent=2))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
