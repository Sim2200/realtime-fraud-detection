"""Model efficiency: ONNX export, int8 quantisation, Core ML, and single-transaction latency.

Inference budget: p95 <= 10 ms per transaction on one CPU thread, measured as
batch-1 calls (the real-time path scores one event at a time). Reported for:
  tree model  -> ONNX (onnxmltools / skl2onnx), float32
  MLP         -> ONNX float32, ONNX int8 (dynamic quantisation of the Linear ops),
                 Core ML (macOS only; CPU and CPU+GPU compute units)
PR-AUC is re-measured on each exported artifact so any accuracy change is visible.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np

ARTIFACTS = Path("results/artifacts")


def _latency(fn, x: np.ndarray, runs: int = 2000, warmup: int = 200) -> dict:
    rows = [x[i % len(x)][None, :] for i in range(warmup + runs)]
    for r in rows[:warmup]:
        fn(r)
    times = []
    for r in rows[warmup:]:
        t0 = time.perf_counter()
        fn(r)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"p50_ms": round(float(np.percentile(t, 50)), 3), "p95_ms": round(float(np.percentile(t, 95)), 3),
            "p99_ms": round(float(np.percentile(t, 99)), 3)}


def export_tree_onnx(model, n_features: int, path: Path) -> Path:
    from onnxmltools import convert_xgboost, convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    from skl2onnx import convert_sklearn

    init = [("input", FloatTensorType([None, n_features]))]
    name = type(model).__name__
    if name == "XGBClassifier":
        onx = convert_xgboost(model, initial_types=init, target_opset=15)
    elif name == "LGBMClassifier":
        onx = convert_lightgbm(model, initial_types=init, target_opset=15, zipmap=False)
    else:
        onx = convert_sklearn(model, initial_types=init, target_opset=15, options={"zipmap": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    return path


def export_mlp_onnx(module, scaler, n_features: int, path: Path) -> Path:
    """Export scaler + MLP + sigmoid as one ONNX graph taking raw features."""
    import torch
    from torch import nn

    class Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mean = nn.Parameter(torch.tensor(scaler.mean_, dtype=torch.float32), requires_grad=False)
            self.scale = nn.Parameter(torch.tensor(scaler.scale_, dtype=torch.float32), requires_grad=False)
            self.mlp = module

        def forward(self, x):
            return torch.sigmoid(self.mlp((x - self.mean) / self.scale))

    w = Wrapped().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(w, torch.zeros(1, n_features), str(path), input_names=["input"], output_names=["prob"],
                      dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}}, opset_version=15)
    return path


def quantize_onnx_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def onnx_session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_scores(path: Path, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    sess = onnx_session(path)
    out_name = sess.get_outputs()[-1].name
    outs = []
    for i in range(0, len(x), batch):
        o = sess.run([out_name], {"input": x[i:i + batch].astype(np.float32)})[0]
        outs.append(o[:, 1] if o.ndim == 2 and o.shape[1] == 2 else o.reshape(-1))
    return np.concatenate(outs)


def benchmark_onnx(path: Path, x: np.ndarray) -> dict:
    sess = onnx_session(path)
    out_name = sess.get_outputs()[-1].name
    lat = _latency(lambda r: sess.run([out_name], {"input": r.astype(np.float32)}), x)
    return {"size_kb": round(path.stat().st_size / 1024, 1), **lat}


def export_coreml(onnx_mlp_module, scaler, n_features: int, path: Path) -> Path | None:
    """Core ML program from the same wrapped MLP (torch.jit trace). macOS only."""
    if platform.system() != "Darwin":
        return None
    import coremltools as ct
    import torch
    from torch import nn

    class Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mean = nn.Parameter(torch.tensor(scaler.mean_, dtype=torch.float32), requires_grad=False)
            self.scale = nn.Parameter(torch.tensor(scaler.scale_, dtype=torch.float32), requires_grad=False)
            self.mlp = onnx_mlp_module

        def forward(self, x):
            return torch.sigmoid(self.mlp((x - self.mean) / self.scale))

    traced = torch.jit.trace(Wrapped().eval(), torch.zeros(1, n_features))
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name="input", shape=(1, n_features))],
                         convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13)
    path = path.with_suffix(".mlpackage")
    mlmodel.save(str(path))
    return path


def benchmark_coreml(path: Path, x: np.ndarray, compute: str = "CPU_ONLY") -> dict:
    import coremltools as ct

    units = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "ALL": ct.ComputeUnit.ALL}[compute]
    model = ct.models.MLModel(str(path), compute_units=units)
    lat = _latency(lambda r: model.predict({"input": r.astype(np.float32)}), x, runs=1000, warmup=100)
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return {"compute_units": compute, "size_kb": round(size / 1024, 1), **lat}


def coreml_scores(path: Path, x: np.ndarray) -> np.ndarray:
    import coremltools as ct

    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    out_key = None
    scores = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        out = model.predict({"input": x[i:i + 1].astype(np.float32)})
        out_key = out_key or next(iter(out))
        scores[i] = np.asarray(out[out_key]).reshape(-1)[0]
    return scores


def hardware() -> str:
    import subprocess
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = platform.processor()
    return f"{cpu} · {platform.system()} {platform.mac_ver()[0] or platform.release()}"
