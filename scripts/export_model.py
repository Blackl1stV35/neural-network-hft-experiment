"""Export a trained PyTorch model to ONNX for production inference.

Usage:
    python scripts/export_model.py --model cnn_lstm --checkpoint models/cnn_lstm_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger

from src.inference.onnx_engine import ONNXExporter, ONNXInferenceEngine
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", choices=["cnn_lstm", "mamba_ssm", "tcn"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="exports/best_model.onnx")
    parser.add_argument("--seq-length", type=int, default=120)
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    args = parser.parse_args()

    setup_logger()

    # Load model
    if args.model == "cnn_lstm":
        from src.models.cnn_lstm import CNNLSTM
        model = CNNLSTM(input_dim=args.feature_dim)
    elif args.model == "mamba_ssm":
        from src.models.mamba_encoder import MambaSSMModel
        model = MambaSSMModel(input_dim=args.feature_dim)
    elif args.model == "tcn":
        from src.models.tcn import TCN
        model = TCN(input_dim=args.feature_dim)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"Loaded model from {args.checkpoint}")

    # Export
    input_shapes = {"x": (1, args.seq_length, args.feature_dim)}
    onnx_path = ONNXExporter.export(model, input_shapes, args.output)
    logger.info(f"ONNX model saved to: {onnx_path}")

    # Benchmark
    if args.benchmark:
        engine = ONNXInferenceEngine(onnx_path, device="cpu", n_threads=4)
        sample = np.random.randn(1, args.seq_length, args.feature_dim).astype(np.float32)
        results = engine.benchmark(n_runs=1000, x=sample)
        logger.info(f"Benchmark results: {results}")


if __name__ == "__main__":
    main()
