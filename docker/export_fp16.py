"""Export YOLOv8m to ONNX in FP16, straight from PyTorch (run during the GPU image build).

    python export_fp16.py /app/models/yolov8m.pt /app/models/yolov8m-fp16.onnx

The model runs in FP16 inside the graph; the input is cast from float32 at the start and the
output back to float32 at the end, so the runtime binds float32 buffers for both models. It's
the same FP16 model the PyTorch FP16 row runs.

Why not convert yolov8m.onnx with onnxconverter-common: the converted graph has mismatched
cast types (Resize, then the Detect head's Cast nodes) and ONNX Runtime refuses to load it.
Tracing FP16 on the build's emulated CPU is slow (about 18 minutes on Apple silicon), but the
layer is cached and only rebuilds when the weights or the lockfile change.
"""

import sys

import onnxruntime as ort
import torch
from ultralytics import YOLO


class Fp16(torch.nn.Module):
    """float32 in -> FP16 model -> float32 (B, 84, 8400) out."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x.half())
        return (out[0] if isinstance(out, list | tuple) else out).float()


def main(weights: str, out: str) -> None:
    model = YOLO(weights).model.fuse().eval()
    for m in model.modules():  # the flags Ultralytics' own exporter sets on the head
        if m.__class__.__name__ == "Detect":
            m.dynamic, m.export, m.format = True, True, "onnx"
    torch.onnx.export(
        Fp16(model),
        torch.zeros(1, 3, 640, 640),
        out,
        opset_version=17,
        dynamo=False,
        input_names=["images"],  # same names as the FP32 export
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
    )
    ort.InferenceSession(out, providers=["CPUExecutionProvider"])  # fail the build if it can't load
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
