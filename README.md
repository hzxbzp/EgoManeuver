# EgoManeuver

Code for **EgoManeuver**, a benchmark for recognizing fine-grained ego-vehicle actions from front-camera video, and
for **DenseEgo**, its baseline model.

EgoManeuver unifies seven public driving datasets (HDD, BDD-X, Argoverse 2, nuScenes, ROAD-Waymo, nuPlan and
PhysicalAI-AV) into one taxonomy of 16 lateral actions (lane keeping, turns, U-turn, lane changes, merges, branches,
nudges, borrowing the adjacent lane, pulling over and out) and 6 longitudinal actions (stationary, start, stop,
accelerate, decelerate, cruise). The labels are derived from the evidence each dataset provides (human annotations,
HD maps with ego poses, captions, ego poses alone); at test time a model sees the video only.

This repository rebuilds every label and segment of the benchmark from the original datasets, trains and evaluates
DenseEgo and the compared video models, and contains a small video-retrieval agent built on DenseEgo.
**No video, sensor, map, label data or model weights are included.** Obtain each dataset from its provider under
its license; the trained weights are obtained by running the training steps below.

```
egokin/   library: taxonomy, dataset readers, label rules (kinematics, HD-map lane matching, pose rules),
          V-JEPA 2 loading with RoPE position interpolation for wide inputs
data/     building clips, labels and benchmark segments
model/    DenseEgo, existing video models, evaluation, transfer benchmarks, probes
agent/    video retrieval agent with and without DenseEgo
lists/    manual correction lists applied by the label pipeline (copy into $VEGO_DATA)
tests/    unit tests (CPU)
```

## Setup

```bash
pip install -r requirements.txt     # Python 3.11, CUDA 12.8
source env.sh                       # sets VEGO_DATA (default ./vego_data), PYTHONPATH, HF_HOME
cp -r lists/* $VEGO_DATA/            # benchmark/*.csv and verify/*.csv
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests
```

Raw datasets go to `$VEGO_DATA/raw/<name>`: HDD (request from Honda Research Institute), BDD100K videos with the
BDD-X annotations, Argoverse 2 Sensor, nuScenes (with the map expansion and CAN bus), nuPlan (sensor blobs and maps),
Waymo Open Dataset with the ROAD-Waymo annotations, and PhysicalAI-AV (training only). Every builder writes
`$VEGO_DATA/clips/<dataset>/<clip_id>/{video.mp4, labels.npz, meta.json}`: 10 fps, 512x288 video and 10 Hz labels
aligned frame by frame.

## 1. Build the benchmark (`data/`)

| Step | Command |
|---|---|
| Clips and per-source labels | `python data/build_hdd_videos.py`, `python data/build_hdd_labels.py`, `python data/build_bdd.py --subset bddx` then `python data/fix_bdd_rotation.py`, `python data/build_av2.py`, `python data/build_nuscenes.py`, `python data/build_nuplan.py --split train --archives 0,1,...` (and the val split), `python data/build_road_waymo.py`, `python data/mine_pai.py` then `python data/fetch_pai_clips.py --budget 20000` |
| Road-relative state (map sources) | `python data/add_road_state.py` |
| Unified temporal boundaries (`lat_u1`) | `python data/relabel_boundaries.py` |
| Rule fixes and verified test-pool events (`lat_u2`) | `python data/relabel_u2.py` (reads `verify/events.csv`, `verify/review_assistant.csv`) |
| Segments | `python data/build_segments.py --overrides $VEGO_DATA/benchmark/test_corrections.csv --out $VEGO_DATA/benchmark/segments_v2.csv` |
| Final benchmark | `python data/finalize_benchmark.py` -> `$VEGO_DATA/benchmark/segments_v2_2.csv` (video-motion check, expert review, test-label audit) |
| Statistics | `python data/benchmark_stats.py --segments segments_v2_2` |

A segment is identified by `dataset|clip_id|start|end` (seconds, one decimal). The correction lists in `lists/`
record every manual change of the quality control: `source` is `expert` (expert review), `assistant` (review of test-pool events by an AI assistant during
development) or `expert_audit` (random audit of the final test labels).

## 2. Train DenseEgo (how to obtain the weights)

DenseEgo fine-tunes the V-JEPA 2 ViT-L encoder (`facebook/vjepa2-vitl-fpc64-256`, downloaded from Hugging Face on
first use) in two stages. Stage 1 trains on the dense per-step labels of 25.6 s windows cut from the full training
clips; stage 2 freezes the encoder and trains a segment head on its cached features. We trained stage 1 on two
NVIDIA B200 GPUs (about 7 h, bf16 autocast, gradient checkpointing); every other step runs on one GPU.

```bash
# 0. frozen V-JEPA 2 features (used by the stage-1 head warm-up and the frozen references)
python model/cache_features.py --model vjepa2x --rope interp --in-scope \
    --datasets pai,hdd,bdd,nuscenes,av2,road_waymo,nuplan --batch 8

# 1. stage 1: head warm-up on frozen features (6,000 steps), then full fine-tuning (10,000 steps),
#    training split only; writes $VEGO_DATA/runs/e2e_off/densego/{head_frozen.pt, ckpt.pt}
torchrun --nproc_per_node 2 model/train_e2e_offline.py --tag densego --backbone vjepa2 --tokens grid33 --tok-pos \
    --L 128 --bs 4 --adapt full --grad-ckpt --warm-steps 6000 --steps 10000 --seed 0 --exclude-val

# 2. cache the step features of the fine-tuned encoder ($VEGO_DATA/cache/feats/ft_densego, packed shards)
for w in 0 1 2 3; do python model/cache_finetuned.py --run densego --shard $w/4; done

# 3. stage 2: segment heads, three seeds; writes $VEGO_DATA/runs/segments/densego_s<seed>_{head.pt,pred.npz}
for s in 0 1 2; do python model/train_segments.py --feats ft_densego --segments segments_v2_2 --tag densego --seed $s; done
```

The trained model consists of `runs/e2e_off/densego/ckpt.pt` (fine-tuned encoder and stage-1 per-step head) and the
stage-2 heads `runs/segments/densego_s<seed>_head.pt`. Main options: learning rate 5e-5 for the last encoder block
with layer-wise decay 0.8 and 1e-4 for the heads (AdamW, weight decay 0.05, one-cycle schedule), 33 tokens per 0.2 s
step (a global token and a 4x8 grid), class-balanced sampling of segments in stage 2.

Check the license of every dataset before sharing trained weights. To train without a dataset, leave it out of
`--datasets` in steps 0-2 (e.g. `--datasets hdd,bdd,av2,nuscenes,road_waymo,nuplan` to train stage 1 without
PhysicalAI-AV) and drop its rows from the segment table for stage 3:

```bash
python -c "import pandas as pd, os; p=os.environ['VEGO_DATA']+'/benchmark/segments_v2_2'; d=pd.read_csv(p+'.csv', dtype=str, keep_default_na=False); d[d.dataset!='pai'].to_csv(p+'_nopai.csv', index=False)"
# then: --segments segments_v2_2_nopai
```

## 3. Evaluate and compare

| Experiment | Command |
|---|---|
| Benchmark evaluation (test) | `python model/benchmark_eval.py --runs densego,<tag>,... --segments segments_v2_2 --part test --boot 1000 --out eval.json` (mAP, rare-class k/n, macro-F1, false alarms on straight, action-family AP; drive-level bootstrap, paired intervals against the first run) |
| Validation predictions | `python model/predict_segments.py --tags densego,... --part val`, then `benchmark_eval.py --runs densego_val,... --part val` |
| Existing video models | `torchrun --nproc_per_node 2 model/train_segments_video.py --model {tsn_r50,i3d_r50,x3d_m,slowfast_r50,timesformer_b,videomae_b} --steps 4000 --bs 16 --segments segments_v2_2 --tag <tag> --seed <s>` |
| Frozen V-JEPA 2 references | `python model/train_segments.py --feats vjepa2x_rinterp --tokens {grid33,pool9} --segments segments_v2_2 --tag <tag> --seed <s>` |
| Control: segment labels only | `torchrun --nproc_per_node 2 model/train_segments_e2e.py --tag segonly --steps 10000 --seed 0`, then `cache_finetuned.py --run segonly` and `train_segments.py --feats ft_segonly` |
| Head variants | `train_segments.py ... --federated`, `--drop-partial`, `--balance 0` |
| HDD (online / offline) | `python model/cache_finetuned.py --run densego --datasets hdd --causal --shard w/4`; `python model/train_hdd_bench.py --aux-kin --bottleneck --tok-pos --steps 8000 --feats ft_causal_densego --causal --window 64 --tag hdd_online --seed <s>` (offline: `--window 128` without `--causal`, `--feats ft_densego`) |
| ROAD-Waymo | `python model/bench_road_waymo.py --run densego` |
| Linear probes | `python model/probe_ridge.py --feats vjepa2x_rinterp --tokens pool9,grid33,mid33 --kinds road`; kinematics: `--kinds kin --kin-datasets nuplan,av2,nuscenes` |

V-JEPA 2 is always loaded through `egokin.models.vjepa_rope.patch_rope` (interpolated RoPE positions for the 224x448
input), and the cuDNN attention backend is disabled for the encoder (`egokin.models.vjepa21.no_cudnn_sdpa`).
PhysicalAI-AV is used for training only.

## 4. Video retrieval agent (`agent/`)

```bash
python agent/agent_index.py                        # DenseEgo's per-step predictions on the test clips -> event index
python agent/agent_scene.py --shard k/3            # Qwen3-VL-8B scene tags, one frame every 10 s (k = 0, 1, 2)
python agent/agent_demo.py --queries q1,q2,q3,q4   # plans, retrieval with and without DenseEgo
```

The per-step predictions come from stage 1 (`runs/e2e_off/<tag>/pred_e2e`). All models run locally; no video frame
is sent to an external service.

## License

The code is released for research use. Each dataset remains under its own license and terms of use.
