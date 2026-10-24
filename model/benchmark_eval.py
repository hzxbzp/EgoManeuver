"""Benchmark evaluation of saved segment predictions (<tag>_s<seed>_pred.npz of train_segments.py /
train_segments_video.py) on one part (test or val) of a benchmark version.

Headline (fixed before the v2 results):
- lateral mAP: moving segments only (NONE is not a lateral class); STRAIGHT is background (a negative for every class,
  not averaged); mean AP over classes with >= MIN_HEAD test positives;
- longitudinal mAP: STATIONARY not averaged; mean AP over classes with >= MIN_HEAD test positives;
- partial labels: inside their set not scored, outside their set negatives (train_segments.ap_table).
Auxiliary: macro-F1 at argmax (headline classes + the background class), STRAIGHT false-alarm rate (straight segments
predicted as a maneuver), per-class accuracy k/n with Clopper-Pearson 95% intervals for classes with fewer than MIN_HEAD
test positives and their mean.
Breakdowns: behaviour families (AP of the summed member probabilities), mistake severity (left <-> right, wrong family),
long-tail groups by TRAINING counts (> 100 / 20-100 / < 20), per dataset (HDD: headline; others: mean recall over their
classes with >= MIN_HEAD positives).
Statistics: mean +- std over seeds; 95% intervals from a bootstrap that resamples whole drives / sessions within each
dataset (the same resamples for all runs, so differences to the first run are paired).

Usage: python model/benchmark_eval.py --runs v2_ours_ft,v2_videomae_b,... [--segments segments_v2] [--part test]
       [--boot 1000] [--out $VEGO_DATA/runs/segments/eval_v2_test.json]
"""

import argparse
import glob
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
from scipy.stats import beta
from sklearn.metrics import f1_score

spec = importlib.util.spec_from_file_location("seg", script_path("train_segments.py"))
seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seg)
LAT, LONG = seg.LAT, seg.LONG
MIN_HEAD = 20
FAMILIES = {"turn": ["TURN_LEFT", "TURN_RIGHT", "U_TURN"],
            "lane_level": ["LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "MERGE_LEFT", "MERGE_RIGHT", "BRANCH_LEFT", "BRANCH_RIGHT"],
            "obstacle": ["NUDGE_LEFT", "NUDGE_RIGHT", "BORROW_LEFT"],
            "curb": ["PULL_OVER", "PULL_OUT"],
            "lane_keeping": ["STRAIGHT", "CURVE"]}
FAMILY_OF = {c: f for f, cs in FAMILIES.items() for c in cs}
MIRROR = {c: c.replace("LEFT", "RIGHT") if c.endswith("LEFT") else c.replace("RIGHT", "LEFT") for c in LAT
          if c.endswith("LEFT") or c.endswith("RIGHT")}


def clopper_pearson(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


class Task:
    """One axis (lateral or longitudinal) of one test part: labels, allowed sets and the class groups."""

    def __init__(self, labels, classes, background, excluded, train_counts):
        self.classes = classes
        self.allowed = seg.target_sets(labels, classes)
        self.exact = self.allowed.sum(1) == 1
        self.y = self.allowed.argmax(1)
        n = {c: int((self.allowed[:, j] & self.exact).sum()) for j, c in enumerate(classes)}
        self.n = n
        self.head = [c for c in classes if n[c] >= MIN_HEAD and c not in excluded + [background]]
        self.rare = [c for c in classes if 0 < n[c] < MIN_HEAD and c not in excluded + [background]]
        self.background = background
        self.groups = {g: [c for c in classes if c not in excluded + [background] and n[c] > 0 and lo <= train_counts.get(c, 0) < hi]
                       for g, (lo, hi) in (("many", (101, 10 ** 9)), ("medium", (20, 101)), ("few", (0, 20)))}

    def ap(self, P, rows, c):
        j = self.classes.index(c)
        al, ex = self.allowed[rows], self.exact[rows]
        pos = al[:, j] & ex
        valid = ex | ~al[:, j]
        if not pos.sum() or not (valid & ~pos).sum():
            return np.nan
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(pos[valid], P[rows][valid, j]))

    def headline(self, P, rows):
        return float(np.nanmean([self.ap(P, rows, c) for c in self.head]))


def lateral_extras(t, P, rows):
    """Auxiliary numbers and breakdowns of the lateral axis (single seed)."""
    ex = rows[t.exact[rows]]
    y, pred = t.y[ex], P[ex].argmax(1)
    names = np.array(LAT)
    out = {}
    lab = [LAT.index(c) for c in t.head + [t.background]]
    out["macro_f1"] = float(f1_score(y, pred, labels=lab, average="macro", zero_division=0))
    st = y == LAT.index("STRAIGHT")
    out["straight_false_alarm"] = float(np.isin(names[pred[st]], ["STRAIGHT", "CURVE", "NONE"], invert=True).mean())
    out["rare"] = {}
    for c in t.rare:
        m = y == LAT.index(c)
        k, n = int((pred[m] == LAT.index(c)).sum()), int(m.sum())
        out["rare"][c] = {"k": k, "n": n, "acc": k / n, "ci": clopper_pearson(k, n)}
    out["rare_mean_acc"] = float(np.mean([v["acc"] for v in out["rare"].values()])) if out["rare"] else np.nan
    fam = {}
    for f, cs in FAMILIES.items():
        ix = [LAT.index(c) for c in cs]
        member = t.allowed[rows][:, ix].any(1)
        inside = t.allowed[rows][:, ix].sum(1) == t.allowed[rows].sum(1)  # allowed set contained in the family
        pos, valid = inside, inside | ~member
        if pos.sum() and (valid & ~pos).sum():
            from sklearn.metrics import average_precision_score
            fam[f] = float(average_precision_score(pos[valid], P[rows][valid][:, ix].sum(1)))
    out["family_ap"] = fam
    man = np.array([FAMILY_OF.get(names[i], "") not in ("", "lane_keeping") for i in y])
    pn, yn = names[pred[man]], names[y[man]]
    out["wrong_direction"] = float(np.mean([p == MIRROR.get(q) for p, q in zip(pn, yn)])) if man.any() else np.nan
    out["wrong_family"] = float(np.mean([FAMILY_OF.get(p, "") != FAMILY_OF.get(q) for p, q in zip(pn, yn)])) if man.any() else np.nan
    out["groups"] = {}
    for g, cs in t.groups.items():
        aps = [t.ap(P, rows, c) for c in cs if t.n[c] >= 5]
        accs = [float((pred[y == LAT.index(c)] == LAT.index(c)).mean()) for c in cs if (y == LAT.index(c)).any()]
        out["groups"][g] = {"classes": cs, "mAP": float(np.nanmean(aps)) if aps else np.nan,
                            "mean_acc": float(np.mean(accs)) if accs else np.nan}
    return out


def longitudinal_extras(t, P, rows):
    ex = rows[t.exact[rows]]
    y, pred = t.y[ex], P[ex].argmax(1)
    lab = [LONG.index(c) for c in t.head + [t.background]]
    out = {"macro_f1": float(f1_score(y, pred, labels=lab, average="macro", zero_division=0)), "rare": {}}
    for c in t.rare:
        m = y == LONG.index(c)
        k, n = int((pred[m] == LONG.index(c)).sum()), int(m.sum())
        out["rare"][c] = {"k": k, "n": n, "acc": k / n, "ci": clopper_pearson(k, n)}
    return out


def per_dataset(t, P, rows, ds):
    out = {}
    names = np.array(LAT)
    for d in sorted(set(ds[rows])):
        r = rows[ds[rows] == d]
        ex = r[t.exact[r]]
        y, pred = t.y[ex], P[ex].argmax(1)
        core = [c for c in LAT if c != "NONE" and (y == LAT.index(c)).sum() >= MIN_HEAD]
        rec = {c: float((pred[y == LAT.index(c)] == LAT.index(c)).mean()) for c in core}
        out[d] = {"core_recall": rec, "mean_recall": float(np.mean(list(rec.values()))) if rec else np.nan}
        if d == "hdd":
            out[d]["lateral_mAP"] = t.headline(P, r)
    return out


def load_run(tag, keys, segments):
    """[(seed, P_lat, P_long)] aligned to keys (rows missing from a run stay NaN and are reported)."""
    out = []
    for f in sorted(glob.glob(f"{seg.ROOT}/runs/segments/{tag}_s[0-9]_pred.npz")):
        z = np.load(f, allow_pickle=True)
        if "key" in z:
            k = z["key"]
        else:  # runs saved before segment keys: rebuild their test order
            args = json.load(open(f.replace("_pred.npz", ".json")))["args"]
            d = seg.load_segments(args.get("feats"), args.get("segments", "segments_v1"))
            k = seg.seg_keys(d[d.part == "test"].reset_index(drop=True))
        lc = list(z["long_classes"]) if "long_classes" in z else ["STATIONARY", "START", "STOP", "ACCELERATE", "DECELERATE",
                                                                  "CRUISE", "DIP", "REVERSE"]
        pos = {x: i for i, x in enumerate(k)}
        ix = np.array([pos.get(x, -1) for x in keys])
        PL = np.full((len(keys), len(LAT)), np.nan, np.float32)
        PO = np.full((len(keys), len(LONG)), np.nan, np.float32)
        ok = ix >= 0
        PL[ok] = z["lat"].astype(np.float32)[ix[ok]]
        lo = z["long"].astype(np.float32)[ix[ok]]
        for j, c in enumerate(LONG):  # classes absent from an old run (none) stay 0; DIP of old runs is dropped
            PO[ok, j] = lo[:, lc.index(c)] if c in lc else 0.0
        out.append((os.path.basename(f)[:-9], PL, PO, ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--segments", default="segments_v2")
    ap.add_argument("--part", default="test")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = seg.load_segments(None, a.segments)
    train_lat = d[d.part == "train"].lat.value_counts().to_dict()
    te = d[d.part == a.part].reset_index(drop=True)
    if "DIP" not in LONG:  # older benchmark versions: slow-down-then-speed-up is not a class any more
        te["long"] = te.long.replace({"DIP": "MOVING"})
    keys = seg.seg_keys(te)
    ds = te.dataset.to_numpy()
    moving = (te.lat != "NONE").to_numpy()
    lat_t = Task(te.lat.tolist(), LAT, "STRAIGHT", ["NONE"], train_lat)
    long_t = Task(te.long.tolist(), LONG, "STATIONARY", [], d[d.part == "train"].long.value_counts().to_dict())
    print(f"{a.segments} {a.part}: {len(te)} segments; lateral headline {lat_t.head}; rare {lat_t.rare}; "
          f"longitudinal headline {long_t.head}; rare {long_t.rare}")
    # cluster bootstrap: whole drives / sessions (clip ids) resampled within each dataset
    rng = np.random.default_rng(0)
    clusters = {}
    for i, (x, c) in enumerate(zip(te.dataset, te.clip_id)):
        clusters.setdefault(x, {}).setdefault(c, []).append(i)
    reps = []
    for _ in range(a.boot):
        r = []
        for x, cl in clusters.items():
            ids = list(cl.values())
            for j in rng.integers(0, len(ids), len(ids)):
                r += ids[j]
        reps.append(np.array(r))
    res, boot = {}, {}
    for tag in a.runs.split(","):
        runs = load_run(tag, keys, a.segments)
        if not runs:
            print(f"{tag}: no predictions")
            continue
        per_seed = []
        for name, PL, PO, ok in runs:
            rows = np.flatnonzero(ok)
            lr = rows[moving[rows]]
            per_seed.append({"seed": name, "coverage": float(ok.mean()), "lateral_mAP": lat_t.headline(PL, lr),
                             "longitudinal_mAP": long_t.headline(PO, rows),
                             "lateral": lateral_extras(lat_t, PL, lr), "longitudinal": longitudinal_extras(long_t, PO, rows),
                             "per_dataset": per_dataset(lat_t, PL, lr, ds),
                             "lateral_AP": {c: lat_t.ap(PL, lr, c) for c in LAT if lat_t.n[c]},
                             "longitudinal_AP": {c: long_t.ap(PO, rows, c) for c in LONG if long_t.n[c]}})
        bl, bo = [], []
        for r in reps:
            sl, so = [], []
            for _, PL, PO, ok in runs:
                rr = r[ok[r]]
                sl.append(lat_t.headline(PL, rr[moving[rr]]))
                so.append(long_t.headline(PO, rr))
            bl.append(np.mean(sl)); bo.append(np.mean(so))
        boot[tag] = (np.array(bl), np.array(bo))
        agg = lambda k: (float(np.mean([s[k] for s in per_seed])), float(np.std([s[k] for s in per_seed])))
        res[tag] = {"seeds": per_seed, "lateral_mAP": agg("lateral_mAP"), "longitudinal_mAP": agg("longitudinal_mAP"),
                    "lateral_mAP_ci": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
                    "longitudinal_mAP_ci": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]}
        s0 = per_seed
        m = lambda f: float(np.nanmean([f(s) for s in s0]))
        print(f"{tag:22s} lat {res[tag]['lateral_mAP'][0] * 100:.1f}+-{res[tag]['lateral_mAP'][1] * 100:.1f} "
              f"[{res[tag]['lateral_mAP_ci'][0] * 100:.1f},{res[tag]['lateral_mAP_ci'][1] * 100:.1f}] | "
              f"long {res[tag]['longitudinal_mAP'][0] * 100:.1f}+-{res[tag]['longitudinal_mAP'][1] * 100:.1f} | "
              f"F1 lat {m(lambda s: s['lateral']['macro_f1']) * 100:.1f} long {m(lambda s: s['longitudinal']['macro_f1']) * 100:.1f} | "
              f"straight FA {m(lambda s: s['lateral']['straight_false_alarm']) * 100:.1f}% | "
              f"rare acc {m(lambda s: s['lateral']['rare_mean_acc']) * 100:.1f} | "
              f"wrong dir {m(lambda s: s['lateral']['wrong_direction']) * 100:.1f}% fam {m(lambda s: s['lateral']['wrong_family']) * 100:.1f}% | "
              f"cov {m(lambda s: s['coverage']):.3f}", flush=True)
    first = a.runs.split(",")[0]
    if first in boot:
        for tag in boot:
            if tag == first:
                continue
            dl, do = boot[first][0] - boot[tag][0], boot[first][1] - boot[tag][1]
            res[tag]["diff_to_" + first] = {"lateral": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                                            "longitudinal": [float(np.percentile(do, 2.5)), float(np.percentile(do, 97.5))]}
            print(f"  {first} - {tag}: lateral [{np.percentile(dl, 2.5) * 100:+.1f}, {np.percentile(dl, 97.5) * 100:+.1f}], "
                  f"longitudinal [{np.percentile(do, 2.5) * 100:+.1f}, {np.percentile(do, 97.5) * 100:+.1f}]")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
