"""Causal sliding-window helpers. Decoder has no access to cues or labels."""
from collections import Counter, defaultdict
import numpy as np

FS = 2000
STEP = 200  # 100ms
REST = 10


def split_runs(record, split):
    """Consecutive held-out contexts, never reset at boundaries within a run.

Offline split boundaries are recording cuts, not speech boundaries. Resetting
there prevents a training-side window/decoder state entering a held-out run.
"""
    runs = []
    for t in record['annotations']:
        if t['split'] != split:
            continue
        if runs and runs[-1][1] == t['prepare']:
            runs[-1][1] = t['context_end']
        else:
            runs.append([t['prepare'], t['context_end']])
    return runs


def window_ends(record, start, end, length):
    # Clock grid depends only on recording time, not on action cue timing.
    first = record['first'] + ((start + length - record['first'] + STEP - 1)//STEP)*STEP
    return range(first, end + 1, STEP)


def training_label(t, start, end, classes):
    """Cue-proxy supervision only; ambiguous transitions are NOT labelled rest."""
    if start < t['prepare'] or end > t['context_end']:
        return None
    overlap = max(0, min(end, t['rest']) - max(start, t['action']))
    # Includes some partial-command windows, but guards earliest reaction period.
    if overlap >= .75*(end-start) and t['action']+300 <= end <= t['rest']:
        return classes.index(t['word'])
    # Conservative pure cue-rest/preparation, with 300ms after-action guard.
    if (start >= t['prepare']+400 and end <= t['action']-400 or
            start >= t['rest']+600 and end <= t['context_end']-200):
        return REST
    return None


def balanced_indices(rows, indices, per_class, seed, replace):
    """Equal classes; balanced subjects, then trial groups, then windows.

Training samples with replacement and changes each epoch. Validation/calibration
round-robin through distinct windows, never duplicates a window to fill a quota.
"""
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(11):
        groups = defaultdict(lambda: defaultdict(list))
        for i in indices:
            r = rows[int(i)]
            if r['label'] == label:
                groups[r['subject']][(r['recording'], r['trial_id'])].append(int(i))
        if not groups:
            raise ValueError(f'Missing class {label}')
        subjects = sorted(groups)
        if replace:
            for n in range(per_class):
                subject = subjects[n % len(subjects)]
                trials = list(groups[subject].values())
                group = trials[int(rng.integers(len(trials)))]
                selected.append(group[int(rng.integers(len(group)))])
        else:
            queues = {}
            for subject in subjects:
                trials = list(groups[subject].values())
                rng.shuffle(trials)
                for trial in trials:
                    rng.shuffle(trial)
                queues[subject] = trials
            n = 0
            while n < per_class and queues:
                for subject in list(queues):
                    trials = queues[subject]
                    trial = trials.pop(0)
                    selected.append(trial.pop())
                    n += 1
                    if trial:
                        trials.append(trial)
                    if not trials:
                        del queues[subject]
                    if n == per_class:
                        break
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


class CommandDecoder:
    """Confirm a word then suppress repeats until sustained predicted rest."""
    def __init__(self, probability=.65, confirm=2, rest_frames=2, margin=0.):
        self.threshold = probability
        self.confirm = confirm
        self.rest_frames = rest_frames
        self.margin = margin
        self.armed = False
        self.rest_count = 0
        self.word = None
        self.count = 0
        self.last_end = None
        self.last_emit = -10**12

    def feed(self, end, probabilities):
        # Any actual sample gap invalidates consecutive evidence.
        if self.last_end is not None and end-self.last_end != STEP:
            self.rest_count, self.count, self.word = 0, 0, None
        self.last_end = end
        order = np.argsort(probabilities)
        label = int(order[-1])
        confidence = float(probabilities[label])
        if label == REST and confidence >= .5:
            self.rest_count += 1
            self.word, self.count = None, 0
            if self.rest_count >= self.rest_frames and end-self.last_emit >= FS//2:
                self.armed = True
            return None
        self.rest_count = 0
        gap = confidence-float(probabilities[order[-2]])
        if not self.armed or label == REST or confidence < self.threshold or gap < self.margin:
            self.word, self.count = None, 0
            return None
        self.count = self.count+1 if self.word == label else 1
        self.word = label
        if self.count < self.confirm:
            return None
        self.armed = False
        self.last_emit = end
        self.word, self.count = None, 0
        return dict(emitted=int(end), label=label, probability=confidence)


def decode(trace, config, classes):
    result = []
    decoder = None
    previous = None
    for rec, run, end, prob in zip(trace['recording'], trace['run'], trace['end'], trace['probabilities']):
        key = (str(rec), int(run))
        if key != previous:
            decoder = CommandDecoder(**config)
            previous = key
        emitted = decoder.feed(int(end), prob)
        if emitted:
            result.append(dict(recording=str(rec), run=int(run), **emitted,
                               candidate=classes[emitted['label']]))
    return result


def event_metrics(events, records, split, subject=None):
    """Common emission-time score for both sliding and old gated models.

Response interval is [action, rest+300ms), clipped to trial context. Entire
context counts for strict success. No window votes are counted as successes.
"""
    rows = []
    owned = defaultdict(list)
    record_map = {r['recording']: r for r in records if subject is None or r['subject'] == subject}
    valid_trials = {}
    for record in record_map.values():
        for t in record['annotations']:
            if t['split'] == split:
                valid_trials[(record['recording'], t['trial_id'])] = t
    # Ownership uses only emission time, independent of predicted/correct label.
    for event in events:
        record = record_map.get(event['recording'])
        if record is None:
            continue
        owners = [t for t in record['annotations'] if t['split'] == split and
                  t['prepare'] <= event['emitted'] < t['context_end']]
        if owners:
            owned[(record['recording'], owners[0]['trial_id'])].append(event)
    correct = clean = missed = wrong = duplicates = outside = rest_outputs = total_outputs = 0
    rest_seconds = duration = 0.
    for key, t in valid_trials.items():
        ev = sorted(owned[key], key=lambda e: e['emitted'])
        response = [e for e in ev if t['action'] <= e['emitted'] < min(t['rest']+600, t['context_end'])]
        good = [e for e in response if e['candidate'] == t['word']]
        correct += bool(good)
        clean += len(ev) == 1 and len(good) == 1
        missed += not response
        wrong += sum(e['candidate'] != t['word'] for e in response)
        duplicates += max(0, len(response)-1)
        outside += len(ev)-len(response)
        rest_outputs += sum(t['rest']+600 <= e['emitted'] < t['context_end'] for e in ev)
        total_outputs += len(ev)
        rest_seconds += max(0, t['context_end']-t['rest']-600)/FS
        duration += (t['context_end']-t['prepare'])/FS
        rows.append(dict(recording=key[0], subject=record_map[key[0]]['subject'], trial_id=key[1],
                         word=t['word'], correct=bool(good), clean=len(ev)==1 and len(good)==1,
                         outputs=len(ev), response_outputs=len(response),
                         predictions=';'.join(e['candidate'] for e in ev),
                         first_correct_after_cue_ms=(good[0]['emitted']-t['action'])/2 if good else None))
    n = len(valid_trials)
    return dict(trials=n, correct_unique_trials=int(correct), end_to_end_recall=correct/max(1,n),
                clean_correct_trials=int(clean), clean_success_rate=clean/max(1,n),
                no_response_trials=int(missed), wrong_label_outputs=int(wrong), duplicate_outputs=int(duplicates),
                outside_response_outputs=int(outside), outputs=total_outputs,
                event_precision=correct/max(1,total_outputs), event_f1=2*correct/max(1,n+total_outputs),
                rest_proxy_outputs=int(rest_outputs), scored_rest_minutes=rest_seconds/60,
                rest_proxy_outputs_per_min=rest_outputs/max(1e-9,rest_seconds/60),
                duration_minutes=duration/60), rows
