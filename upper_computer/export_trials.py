"""Export block-marker cue windows; never substitute host receive time for ADC time."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def export(path, allow_simulated=False):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if meta["mode"] == "legacy":
        raise ValueError("旧固件无设备采样序号，禁止按串口接收时间生成训练片段")
    if meta["mode"] == "simulation" and not allow_simulated:
        raise ValueError("模拟数据只能显式使用 --allow-simulated 导出，不能作为真实实验数据")
    if meta["status"] == "recording":
        raise ValueError("记录尚未正常结束，请先检查文件完整性")
    events = load_jsonl(path / "events.jsonl")
    results = {r["trial_id"]: r for r in load_jsonl(path / "trials.jsonl")}
    ack = {e["event_id"]: e for e in events if e["kind"] == "marker_ack"}
    phases = {}
    for e in events:
        if e["kind"] == "phase" and "trial_id" in e:
            phases.setdefault(e["trial_id"], {})[e["phase"]] = e
    samples = np.genfromtxt(path / "samples.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    samples = np.atleast_1d(samples)
    indices = samples["device_sample_idx"].astype(np.int64)
    raw = samples["raw_ch0"].astype(np.uint16)
    out = path / "cue_windows"
    out.mkdir(exist_ok=False)
    manifest = []
    for trial_id, result in results.items():
        item = dict(trial_id=trial_id, word=result["word"], label_id=result["label_id"],
                    exported=False, reason="", samples=0, start_sample="", end_sample="")
        try:
            if result["status"] != "completed":
                raise ValueError(result["status"])
            phase = phases[trial_id]
            start, end = ack[phase["action"]["event_id"]], ack[phase["rest"]["event_id"]]
            if max(start["rtt_ms"], end["rtt_ms"]) > 100:
                raise ValueError("marker_rtt_over_100ms")
            a, b = start["device_sample_idx"], end["device_sample_idx"]
            if b <= a:
                raise ValueError("invalid_or_wrapped_interval")
            mask = (indices >= a) & (indices < b)
            idx, values = indices[mask], raw[mask]
            if len(idx) != b - a or not np.array_equal(idx, np.arange(a, b)):
                raise ValueError("missing_samples")
            # No onset inference and no label-derived preprocessing of the raw signal.
            np.savez_compressed(out / f"trial_{trial_id:04d}_{result['word']}.npz",
                                raw_ch0=values, sample_idx=idx, label_id=result["label_id"],
                                word=result["word"], fs=meta["sampling_rate_hz"], trial_id=trial_id)
            item.update(exported=True, samples=len(idx), start_sample=a, end_sample=b)
        except (KeyError, ValueError) as exc:
            item["reason"] = str(exc)
        manifest.append(item)
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "word", "label_id", "exported", "reason", "samples", "start_sample", "end_sample"])
        writer.writeheader(); writer.writerows(manifest)
    (out / "README.txt").write_text(
        "These are approximate cue windows, NOT measured muscle activity onsets.\n"
        "Device markers use the last completed 20-sample block boundary.\n"
        "UART/USB latency and physical display timing remain uncalibrated.\n"
        "Review waveform/labels before training. Preserve subject/session/trial grouping.\n"
        f"Source mode: {meta['mode']}\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--allow-simulated", action="store_true")
    args = parser.parse_args()
    manifest = export(args.recording, args.allow_simulated)
    print(f"Exported {sum(x['exported'] for x in manifest)}/{len(manifest)} trials. See cue_windows/manifest.csv.")
