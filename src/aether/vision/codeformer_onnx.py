"""Explicit CodeFormer ONNX single-input RGB512 adapter.

Only compatible *wrapped* ONNX exports are accepted: input float32
NCHW [1,3,512,512] RGB in [-1,1], output the same shape/range.
This is a generated face hypothesis, never historical identity evidence.
Face detection, alignment and identity veto are separate responsibilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


class CodeFormerOnnxAdapter:
    SIZE = 512

    def __init__(self, session):
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("CodeFormer ONNX wrapper must expose one input/output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        expected = (1, 3, self.SIZE, self.SIZE)
        for descriptor in (inputs[0], outputs[0]):
            shape = descriptor.shape
            if len(shape) != 4 or any(
                type(value) is int and value != required
                for value, required in zip(shape, expected)
            ):
                raise ValueError("CodeFormer wrapper must use NCHW RGB512")
        self.session = session

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, *,
                        provider: str = "CPUExecutionProvider"):
        checkpoint = Path(checkpoint)
        if checkpoint.suffix.lower() != ".onnx" or not checkpoint.is_file():
            raise FileNotFoundError("Compatible local CodeFormer .onnx required")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install onnxruntime on the chosen execution host") from error
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"ONNX execution provider unavailable: {provider}")
        session = ort.InferenceSession(str(checkpoint), providers=[provider])
        return cls(session)

    def restore_crop(self, rgb512: np.ndarray) -> np.ndarray:
        source = np.asarray(rgb512)
        if source.shape != (self.SIZE, self.SIZE, 3) or source.dtype != np.uint8:
            raise ValueError("A pre-aligned RGB8 face crop of exactly 512x512 is required")
        normalized = (source.astype(np.float32) / 127.5 - 1.0)
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        result = self.session.run([self.output_name], {self.input_name: tensor})
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise ValueError("CodeFormer returned an invalid output collection")
        predicted = np.asarray(result[0])
        if (predicted.shape != (1, 3, self.SIZE, self.SIZE) or
                not np.issubdtype(predicted.dtype, np.floating) or
                not np.isfinite(predicted).all() or
                float(predicted.min()) < -1.001 or
                float(predicted.max()) > 1.001):
            raise ValueError("CodeFormer output must be finite NCHW RGB in [-1,1]")
        rgb = np.clip((predicted[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255)
        return np.rint(rgb).astype(np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether-codeformer",
        description="Run explicitly supplied CodeFormer ONNX export on one aligned face crop",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-crop", type=Path, required=True)
    parser.add_argument("--output-hypothesis", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args(argv)

    source, output = args.input_crop, args.output_hypothesis
    if (not source.is_file() or source.resolve() == output.resolve() or
            output.exists() or output.suffix.lower() != ".png"):
        raise ValueError("Input must exist and a new PNG output path is required")
    with Image.open(source) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
            raise ValueError("Input must be a pre-aligned 512x512 RGB PNG")
        crop = np.asarray(image, dtype=np.uint8).copy()

    engine = CodeFormerOnnxAdapter.from_checkpoint(args.model, provider=args.provider)
    hypothesis = engine.restore_crop(crop)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError("Unresolved CodeFormer partial output")
    Image.fromarray(hypothesis, mode="RGB").save(partial, format="PNG")
    with Image.open(partial) as check:
        if check.format != "PNG" or check.mode != "RGB" or check.size != (512, 512):
            raise RuntimeError("CodeFormer candidate integrity validation failed")
        check.verify()
    os.replace(partial, output)

    report = {
        "status": "executed",
        "output": str(output),
        "output_sha256": _sha256(output),
        "model_sha256": _sha256(args.model),
        "provider_requested": args.provider,
        "generated_hypothesis": True,
        "identity_verified": False,
        "input_crop_dimensions": [512, 512],
        "fidelity_limit": "inference does not establish historical face identity",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
