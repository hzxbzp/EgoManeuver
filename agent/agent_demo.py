"""Retrieval agent: a local language model turns a request for driving video into a plan,
and tools retrieve test-clip moments; run once with DenseEgo's ego-behaviour index and once without it, where the
vision-language model judges the ego behaviour from frames. Qualitative demo only.

Plan (JSON from the language model): {"behaviors": [ego-behaviour classes in temporal order],
"scene": {"road": [...], "intersection": true/false/null, "objects": [...]}, "question": yes/no question on the ego
behaviour for the frame-based judge}.
With DenseEgo: events of agent/ego_events.csv (agent/agent_index.py), chained in order (next event starting at most
CHAIN_S after the previous one ends), filtered by the scene tags of agent/scene/*.jsonl (agent/agent_scene.py) within
SCENE_PAD s, ranked by the mean event score. Without DenseEgo: windows of WIN_S s every STRIDE_S s that pass the same
scene filter (at most MAX_CAND, sampled with a fixed seed), ranked by P(yes) of Qwen3-VL-8B on 8 frames.
Correctness marks: the per-frame labels of the clip (lat_u2, long) contain every requested class, in order, for at
least 0.5 s inside the returned span. Output: agent/demo/<qid>.json and thumbnails agent/demo/thumbs/<qid>_<agent>_<rank>.jpg.

Usage: python agent/agent_demo.py [--queries q1,q2] [--max-cand 800]
"""

import argparse
import glob
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from egokin.data.index import ROOT
from egokin.taxonomy import Lat, Long

CHAIN_S, SCENE_PAD, WIN_S, STRIDE_S, K = 6.0, 5.0, 8.0, 4.0, 5
LAT_CLASSES = ["CURVE", "TURN_LEFT", "TURN_RIGHT", "U_TURN", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "MERGE_LEFT",
               "MERGE_RIGHT", "BRANCH_LEFT", "BRANCH_RIGHT", "PULL_OVER", "PULL_OUT", "NUDGE_LEFT", "NUDGE_RIGHT",
               "BORROW_LEFT"]
LONG_CLASSES = ["STATIONARY", "START", "ACCELERATE", "DECELERATE", "DECEL_TO_STOP"]
QUERIES = {
    "q1": "Find clips where our car changes into the left lane on a highway.",
    "q2": "Show me moments where the car waits at a traffic light and then turns left.",
    "q3": "I need examples of the car swerving around a parked vehicle.",
    "q4": "Find clips where the car pulls over to the side of the road.",
}
SYSTEM = """You plan searches in a library of front-camera driving videos. Turn the user's request into JSON with keys:
"behaviors": the ego-vehicle behaviors the request needs, in temporal order, chosen from
  lateral: CURVE, TURN_LEFT, TURN_RIGHT, U_TURN, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, MERGE_LEFT, MERGE_RIGHT (from an ending
  lane), BRANCH_LEFT, BRANCH_RIGHT (into a lane that splits off, e.g. a turn pocket), PULL_OVER, PULL_OUT (from the curb),
  NUDGE_LEFT, NUDGE_RIGHT (moving aside within the lane), BORROW_LEFT (crossing into the next lane to pass an obstacle);
  longitudinal: STATIONARY, START (from standstill), ACCELERATE, DECELERATE, DECEL_TO_STOP;
  waiting or standing still is STATIONARY and moving off from standstill is START;
  when several behaviors are alternatives at one position, give them as a nested list, e.g. [["NUDGE_LEFT", "BORROW_LEFT"]];
"scene": constraints on what is visible, with optional keys "road" (list of "highway", "urban street", "residential street",
  "rural road", "parking lot"), "intersection" (true/false), "objects" (list of "pedestrian", "cyclist", "parked car",
  "parked truck", "moving truck", "bus", "traffic light", "stop sign", "construction", "tram tracks"; any of them suffices);
"question": one yes/no question about the ego vehicle's behavior in a short video, to check a candidate by watching it.
Answer with the JSON only."""
EXAMPLE = ("Find the car halting at a stop sign before it turns right.",
           '{"behaviors": ["STATIONARY", "START", "TURN_RIGHT"], "scene": {"objects": ["stop sign"]}, '
           '"question": "Does the car stop and then turn right?"}')


EXAMPLE2 = ("Find the car leaving the highway at an exit.",
            '{"behaviors": [["BRANCH_RIGHT", "LANE_CHANGE_RIGHT"]], "scene": {"road": ["highway"]}, '
            '"question": "Does the car leave the highway through an exit lane?"}')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VLM:
    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        name = "Qwen/Qwen3-VL-8B-Instruct"
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]

    def _inputs(self, content):
        msgs = [{"role": "user", "content": content}]
        return self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                             return_tensors="pt").to("cuda")

    @torch.no_grad()
    def chat(self, text, max_new_tokens=200):
        inp = self._inputs([{"type": "text", "text": text}])
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    @torch.no_grad()
    def p_yes(self, images, question):
        inp = self._inputs([{"type": "image", "image": im} for im in images] +
                           [{"type": "text", "text": question + " Answer yes or no."}])
        lg = self.model(**inp).logits[0, -1].float()
        return torch.softmax(torch.stack([lg[self.yes].logsumexp(0), lg[self.no].logsumexp(0)]), 0)[0].item()


def plan(vlm, request):
    text = (f"{SYSTEM}\n\nRequest: {EXAMPLE[0]}\nJSON: {EXAMPLE[1]}\n\nRequest: {EXAMPLE2[0]}\nJSON: {EXAMPLE2[1]}"
            f"\n\nRequest: {request}\nJSON:")
    raw = vlm.chat(text)
    s, e = raw.find("{"), raw.rfind("}")
    p = json.loads(raw[s:e + 1])
    beh = [[x for x in (b if isinstance(b, list) else [b]) if x in LAT_CLASSES + LONG_CLASSES] for b in p.get("behaviors", [])]
    p["behaviors"] = [b for b in beh if b]  # each position: a list of alternative classes
    p.setdefault("question", f"Does this video show the following: {request}")
    return p, raw


def load_scene():
    rows = [json.loads(l) for f in sorted(glob.glob(f"{ROOT}/agent/scene/*of*.jsonl")) if "smoke" not in f for l in open(f)]
    d = pd.DataFrame([r for r in rows if r["tags"]])
    return {k: g.sort_values("t") for k, g in d.groupby(["dataset", "clip_id"])}


def scene_ok(scene, ds, cid, t0, t1, cons):
    """Any keyframe within the padded span satisfies all constraints (objects: any of them)."""
    g = scene.get((ds, cid))
    if g is None or not cons:
        return g is not None or not cons
    g = g[(g.t >= t0 - SCENE_PAD) & (g.t <= t1 + SCENE_PAD)]
    for tags in g.tags:
        ok = True
        if cons.get("road") and tags.get("road") not in cons["road"]:
            ok = False
        if cons.get("intersection") is not None and bool(tags.get("intersection")) != bool(cons["intersection"]):
            ok = False
        if cons.get("objects") and not set(cons["objects"]) & set(tags.get("objects") or []):
            ok = False
        if ok:
            return True
    return False


def chains(ev, behaviors):
    """Ordered chains of events, one per requested behavior; span and mean score."""
    by = [ev[ev.cls.isin(alts)] for alts in behaviors]
    out = []
    for r in by[0].itertuples():
        chain = [(r.dataset, r.clip_id, r.t0, r.t1, r.score)]
        for e in by[1:]:
            last = chain[-1]  # the next behavior starts after the previous one ends (1 s overlap allowed)
            nxt = e[(e.dataset == last[0]) & (e.clip_id == last[1]) & (e.t0 >= last[3] - 1.0) & (e.t0 <= last[3] + CHAIN_S)]
            if not len(nxt):
                chain = None
                break
            n = nxt.sort_values("t0").iloc[0]
            chain.append((n.dataset, n.clip_id, n.t0, n.t1, n.score))
        if chain:
            out.append(dict(dataset=chain[0][0], clip_id=chain[0][1], t0=chain[0][2], t1=chain[-1][3],
                            score=float(np.mean([x[4] for x in chain]))))
    return pd.DataFrame(out)


def diversify(d, k, gap=60.0):
    """Top k, at most one result per clip within `gap` seconds."""
    picked = []
    for r in d.sort_values("score", ascending=False).itertuples():
        if any(p.dataset == r.dataset and p.clip_id == r.clip_id and abs(p.t0 - r.t0) < gap for p in picked):
            continue
        picked.append(r)
        if len(picked) == k:
            break
    return pd.DataFrame([p._asdict() for p in picked]).drop(columns="Index", errors="ignore")


def correct(ds, cid, t0, t1, behaviors):
    """Per-frame labels contain every requested class, in order, for >= 0.5 s inside [t0, t1]."""
    z = np.load(f"{ROOT}/clips/{ds}/{cid}/labels.npz")
    f0, f1 = int(t0 * 10), int(np.ceil(t1 * 10))
    pos = f0
    for alts in behaviors:
        m = np.zeros(f1 - pos, bool)
        for c in alts:
            lab = z["lat_u2"] if c in LAT_CLASSES else z["long"]
            m |= lab[pos:f1] == (int(Lat[c]) if c in LAT_CLASSES else int(Long[c]))
        hit = np.flatnonzero(m)
        if len(hit) < 5:
            return False
        pos = pos + int(hit[0])
    return True


def frames_of(e2e, ds, cid, t0, t1, n):
    f0, f1 = int(t0 * 10), max(int(t0 * 10) + 1, int(t1 * 10) - 1)
    fr = e2e.decode(f"{ROOT}/clips/{ds}/{cid}/video.mp4", f0, f1)
    return [fr[i] for i in np.linspace(0, len(fr) - 1, n).round().astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=",".join(QUERIES))
    ap.add_argument("--max-cand", type=int, default=800)
    a = ap.parse_args()
    from PIL import Image
    e2e = _load("train_e2e")
    vlm = VLM()
    ev = pd.read_csv(f"{ROOT}/agent/ego_events.csv", dtype={"clip_id": str})
    scene = load_scene()
    durations = {k: g.t.max() + 5.0 for k, g in scene.items()}
    out = f"{ROOT}/agent/demo"
    os.makedirs(f"{out}/thumbs", exist_ok=True)
    for qid in a.queries.split(","):
        request = QUERIES[qid]
        p, raw = plan(vlm, request)
        beh, cons = p["behaviors"], p.get("scene") or {}
        res = {"request": request, "plan": p, "plan_raw": raw}
        # with DenseEgo: chained events, scene filter
        t = time.time()
        c = chains(ev, beh) if beh else pd.DataFrame()
        if len(c):
            c = c[[scene_ok(scene, r.dataset, r.clip_id, r.t0, r.t1, cons) for r in c.itertuples()]]
        top = diversify(c, K) if len(c) else pd.DataFrame()
        res["with"] = dict(n_candidates=int(len(c)), seconds=round(time.time() - t, 1), results=top.to_dict("records"))
        # without DenseEgo: sliding windows that pass the scene filter, judged by the VLM on 8 frames
        t = time.time()
        wins = [(ds, cid, s, s + WIN_S) for (ds, cid), dur in durations.items()
                for s in np.arange(0.0, max(dur - WIN_S, 0.0) + 1e-6, STRIDE_S)]
        wins = [w for w in wins if scene_ok(scene, *w, cons)]
        rng = np.random.default_rng(0)
        n_all = len(wins)
        if len(wins) > a.max_cand:
            wins = [wins[i] for i in sorted(rng.choice(len(wins), a.max_cand, replace=False))]
        scored = []
        for ds, cid, t0, t1 in wins:
            ims = [Image.fromarray(f).resize((448, 252)) for f in frames_of(e2e, ds, cid, t0, t1, 8)]
            scored.append(dict(dataset=ds, clip_id=cid, t0=float(t0), t1=float(t1), score=vlm.p_yes(ims, p["question"])))
        top0 = diversify(pd.DataFrame(scored), K) if scored else pd.DataFrame()
        res["without"] = dict(n_windows=n_all, n_judged=len(wins), seconds=round(time.time() - t, 1),
                              results=top0.to_dict("records"))
        for agent in ("with", "without"):
            for rank, r in enumerate(res[agent]["results"]):
                r["correct"] = bool(beh) and correct(r["dataset"], r["clip_id"], r["t0"], r["t1"], beh)
                ims = frames_of(e2e, r["dataset"], r["clip_id"], r["t0"], r["t1"], 3)
                strip = np.concatenate([np.asarray(Image.fromarray(im).resize((256, 144))) for im in ims], 1)
                Image.fromarray(strip).save(f"{out}/thumbs/{qid}_{agent}_{rank}.jpg", quality=90)
        json.dump(res, open(f"{out}/{qid}.json", "w"), indent=1, default=float)
        print(qid, json.dumps({k: res[k] for k in ("request", "plan")}), flush=True)
        for agent in ("with", "without"):
            print(f"  {agent}: " + "; ".join(f"{r['dataset']}/{r['clip_id']} {r['t0']:.1f}-{r['t1']:.1f} "
                                             f"{r['score']:.2f} {'ok' if r['correct'] else 'x'}" for r in res[agent]["results"]),
                  flush=True)
    print("AGENT DONE")


if __name__ == "__main__":
    main()
