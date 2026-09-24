"""Trial scheduling and append-only acquisition files."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import time
import uuid
from . import __version__
from .vocabulary import DEFAULT_WORDS, validate_words

WORDS = DEFAULT_WORDS
LABELS = {word: i + 1 for i, word in enumerate(WORDS)}


def schedule(repetitions, seed, words=WORDS):
    vocabulary = validate_words(words)
    labels = {word: i + 1 for i, word in enumerate(vocabulary)}
    rng = random.Random(seed)
    trials = []
    for block in range(1, repetitions + 1):
        words = list(vocabulary)
        rng.shuffle(words)
        for word in words:
            trials.append(dict(trial_id=len(trials) + 1, word=word,
                               label_id=labels[word], block=block, repeat_of=None))
    return trials


class TrialEngine:
    """Each actual GUI transition starts a full phase; never skip overdue phases."""
    def __init__(self, trials, durations, emit):
        self.trials = [dict(t) for t in trials]
        self.durations = durations
        self.emit = emit
        self.pos = -1
        self.phase = "idle"
        self.deadline = 0
        self.invalid_reason = None
        self.completed = 0
        self.finished = set()

    @property
    def current(self):
        return self.trials[self.pos] if 0 <= self.pos < len(self.trials) else None

    def transition(self, phase, now):
        previous_deadline = self.deadline
        self.phase = phase
        self.deadline = now + int(self.durations.get(phase, 0) * 1e9)
        self.emit(dict(kind="phase", phase=phase, host_ns=now,
                       planned_host_ns=previous_deadline or now,
                       **(self.current or {})))

    def start(self, now):
        self.transition("baseline", now)

    def _next(self, now):
        self.pos += 1
        self.invalid_reason = None
        self.transition("prepare" if self.current else "finished", now)

    def tick(self, now):
        if self.phase in ("idle", "paused", "finished", "stopped") or now < self.deadline:
            return
        if self.phase == "baseline":
            self._next(now)
        elif self.phase == "prepare":
            self.transition("action", now)
        elif self.phase == "action":
            self.transition("rest", now)
        elif self.phase == "rest":
            self.finish_trial(now, "rejected" if self.invalid_reason else "completed",
                              self.invalid_reason)
            self._next(now)

    def finish_trial(self, now, status, reason):
        if self.current and self.current["trial_id"] not in self.finished:
            self.finished.add(self.current["trial_id"])
            self.completed += 1
            self.emit(dict(kind="trial_result", host_ns=now, status=status,
                           reason=reason, **self.current))

    def repeat_current(self):
        if self.current:
            trial = dict(self.current)
            trial["repeat_of"] = trial["trial_id"]
            trial["trial_id"] = max(t["trial_id"] for t in self.trials) + 1
            self.trials.append(trial)

    def reject(self, now):
        if self.current and self.phase in ("prepare", "action", "rest") and not self.invalid_reason:
            self.invalid_reason = "operator_rejected"
            self.repeat_current()
            self.emit(dict(kind="reject", host_ns=now, **self.current))

    def pause(self, now):
        if self.phase not in ("baseline", "prepare", "action", "rest"):
            return
        if self.current:
            if not self.invalid_reason:
                self.repeat_current()
            self.finish_trial(now, "interrupted", "operator_pause")
        self.transition("paused", now)

    def resume(self, now):
        if self.phase == "paused":
            if self.pos < 0:
                self.transition("baseline", now)
            else:
                self._next(now)

    def stop(self, now, reason="operator_stop"):
        self.finish_trial(now, "interrupted", reason)
        self.transition("stopped", now)


class Recorder:
    def __init__(self, root, config):
        words = validate_words(config.get("words", list(WORDS)))
        subject = config["subject_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", subject):
            raise ValueError("受试者编号只允许字母、数字、下划线和短横线")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        group = "simulated" if config["mode"] == "simulation" else "recordings"
        self.path = Path(root) / group / subject / f"session_{config['session_id']:03d}" / stamp
        self.path.mkdir(parents=True, exist_ok=False)
        self.files = {}
        self.rows = 0
        self.last_flush = time.monotonic()
        self.meta = dict(config, schema_version="1.0", app_version=__version__,
                         started_utc=datetime.now(timezone.utc).isoformat(),
                         host_clock="perf_counter_ns", status="recording",
                         signal_unit="ADC counts", channel_count=1, adc_bits=16,
                         condition="silent", articulation_mode="closed_lip",
                         label_map={"0": "rest", **{str(i+1): w for i, w in enumerate(words)}},
                         timing="legacy_unaligned" if config["mode"] == "legacy" else "device_block_marker_20ms_unvalidated_display",
                         raw_data_filtered_during_capture=False)
        self.meta["words"] = words
        try:
            for name in ("events.jsonl", "trials.jsonl", "frames.jsonl"):
                self.files[name] = (self.path / name).open("x", encoding="utf-8", buffering=1)
            self.files["transport.bin"] = (self.path / "transport.bin").open("xb")
            self.files["samples.csv"] = (self.path / "samples.csv").open("x", encoding="utf-8", newline="")
            self.csv = csv.writer(self.files["samples.csv"])
            self.csv.writerow(["host_row_idx", "device_sample_idx", "rx_batch_id", "rx_host_ns", "raw_ch0", "device_filtered", "device_envelope"])
            self.save_meta()
        except Exception:
            for f in self.files.values():
                f.close()
            raise

    def save_meta(self):
        temp = self.path / "metadata.tmp"
        temp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path / "metadata.json")

    def json(self, name, item):
        self.files[name].write(json.dumps(item, ensure_ascii=False) + "\n")

    def samples(self, values, first, batch, rx_ns):
        for i, (raw, filt, env) in enumerate(values):
            self.csv.writerow([self.rows, "" if first is None else (first + i) & 0xFFFFFFFF,
                               batch, rx_ns, raw, filt, env])
            self.rows += 1
        if time.monotonic() - self.last_flush >= 1:
            for f in self.files.values():
                f.flush()
            self.last_flush = time.monotonic()

    def close(self, status, stats, error=None):
        self.meta.update(status=status, samples_saved=self.rows, stats=stats,
                         ended_utc=datetime.now(timezone.utc).isoformat(), error=error)
        try:
            for f in self.files.values():
                f.flush()
            self.save_meta()
        finally:
            for f in self.files.values():
                f.close()
