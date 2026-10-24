"""Scene index for the retrieval agent: one keyframe every STEP_S seconds of every test clip
(the clips of DenseEgo's stage-1 test predictions), described by a local vision-language model (Qwen3-VL-8B-Instruct)
as JSON with a fixed vocabulary. Both agents of the demo (with and without DenseEgo) use this index for the scene.

Output: $VEGO_DATA/agent/scene/<shard>.jsonl, one line per keyframe: dataset, clip_id, t (s), tags (dict) or raw text.

Usage: python agent/agent_scene.py --shard 0/3 [--limit 20]
"""

import argparse
import glob
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import torch

from egokin.data.index import ROOT

STEP_S = 10.0
PROMPT = """This is one frame from the front camera of a car. Describe the scene with JSON only, using exactly these keys and values:
{"road": one of "highway", "urban street", "residential street", "rural road", "parking lot",
 "intersection": true or false (the car is at or inside an intersection),
 "weather": one of "clear", "rain", "snow", "fog",
 "light": one of "day", "dusk", "night",
 "traffic": one of "none", "light", "heavy",
 "objects": a list with any of "pedestrian", "cyclist", "parked car", "parked truck", "moving truck", "bus", "traffic light", "stop sign", "construction", "tram tracks"}"""


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clips(run):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/runs/e2e_off/{run}/pred_e2e/*.npz")):
        ds, cid = os.path.basename(f)[:-4].split("__", 1)
        dur = np.load(f)["pb"].shape[0] * 0.2
        out.append((ds, cid, dur))
    return out


def parse(text):
    s, e = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--run", default="trainonly_full_s0")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N keyframes")
    a = ap.parse_args()
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    e2e = _load("train_e2e")
    k, n = map(int, a.shard.split("/"))
    todo = [(ds, cid, t) for ds, cid, dur in clips(a.run)
            for t in (np.arange(STEP_S / 2, dur, STEP_S) if dur >= STEP_S else [dur / 2])]
    todo = todo[k::n][: a.limit or None]
    out_dir = f"{ROOT}/agent/scene"
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/{k}of{n}{'_smoke' if a.limit else ''}.jsonl"
    done = set()
    if os.path.exists(dst):
        done = {(r["dataset"], r["clip_id"], r["t"]) for r in map(json.loads, open(dst))}
    name = "Qwen/Qwen3-VL-8B-Instruct"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(dst, "a") as fo:
        for i, (ds, cid, t) in enumerate(todo):
            t = round(float(t), 1)
            if (ds, cid, t) in done:
                continue
            f = int(t * 10)
            im = Image.fromarray(e2e.decode(f"{ROOT}/clips/{ds}/{cid}/video.mp4", f, f)[0])
            msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=120, do_sample=False)
            text = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            tags = parse(text)
            fo.write(json.dumps(dict(dataset=ds, clip_id=cid, t=t, tags=tags, raw=None if tags else text)) + "\n")
            fo.flush()
            if i % 200 == 0:
                print(i, len(todo), ds, cid, t, tags, flush=True)
    print("SCENE DONE", dst)


if __name__ == "__main__":
    main()
