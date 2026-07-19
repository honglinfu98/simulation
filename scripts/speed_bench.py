#!/usr/bin/env python3
"""Training-step and sampling-throughput benchmark across the model zoo.

Loads each benchmark checkpoint (btc, seed 1), times
  (a) one training fwd+bwd on the benchmark shape (B=64, N=1024, K=62), and
  (b) closed-loop simulation throughput (events/s) with the benchmark rollout
      protocol (carried state where the decoder has one, 8x60s here),
      inversion sampler for every model, plus Ogata thinning for SS2P2
      (exact ceiling) and the sequential-loop training path for the S2P2
      family as the non-parallel reference.

    PYTHONPATH=. python3 scripts/speed_bench.py --root experiments/ma_cbse/btc \
        --out /tmp/speed_bench.json
"""
import argparse
import json
import time

import torch

from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.evaluation.stylized_facts import (
    simulate_stream, simulate_stream_thinning)

TAGS = ["nhp", "lstm", "sahp", "pct-lstm", "s2p2", "ss2p2-full"]
K = 62


def synth_batch(b, n, device, rate=38.0):
    torch.manual_seed(0)
    marks = torch.zeros(b, n, K, device=device)
    idx = torch.randint(0, K, (b, n), device=device)
    marks.scatter_(2, idx.unsqueeze(-1), 1.0)
    dts = torch.rand(b, n, device=device).mul_(2.0 / rate)
    return {"input_marks": marks, "input_times": dts,
            "target_time": dts[:, -1].clone(), "target_marks": marks[:, -1].clone()}


def time_train_step(model, batch, device, reps=5):
    model.train()
    for _ in range(2):
        loss, _ = model.compute_loss(batch, device)
        loss.backward(); model.zero_grad()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        loss, _ = model.compute_loss(batch, device)
        loss.backward(); model.zero_grad()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def time_rollout(model, batch, device, sampler, carried, duration=60.0, n_seq=8):
    model.eval()
    torch.cuda.synchronize(); t0 = time.time()
    if sampler == "thinning":
        m, d, c = simulate_stream_thinning(model, batch, device, 0, n_seq, 1,
                                           duration=duration, carried=carried)
    else:
        m, d, c = simulate_stream(model, batch, device, 0, n_seq, 10.0, 64, 1,
                                  duration=duration, carried=carried)
    torch.cuda.synchronize()
    wall = time.time() - t0
    n_ev = int(sum((c[i] <= duration).sum() for i in range(m.shape[0])))
    return n_ev / wall, wall, n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    train_b = synth_batch(64, 1024, device)
    roll_b = synth_batch(8, 1024, device)
    out = {"gpu": torch.cuda.get_device_name(0)}
    for tag in TAGS:
        ck = torch.load(f"{args.root}/{tag}-s1/train/best_model.pt",
                        map_location=device, weights_only=False)
        cfg = ck["config"]
        r = {}
        variants = [("train_s", dict())]
        if tag in ("s2p2", "ss2p2-full"):
            variants = [("train_scan_s", dict(s2p2_scan=True)),
                        ("train_loop_s", dict(s2p2_scan=False))]
        for name, over in variants:
            cfg2 = dict(cfg); cfg2.update(over)
            model = create_volume_set_mtpp(K, cfg2, device,
                                           use_volume=cfg.get("use_volume", True),
                                           intensity_type=cfg.get("intensity_type", "dynamic"))
            model.load_state_dict(ck["model_state_dict"]); model.to(device)
            r[name] = time_train_step(model, train_b, device)
        # sampling: carried where the decoder has a Markov packed state
        carried = tag != "sahp"
        evs, wall, n = time_rollout(model, roll_b, device, "inversion", carried)
        r["sim_inversion_ev_s"], r["sim_inversion_events"] = evs, n
        if tag == "ss2p2-full":
            evs, wall, n = time_rollout(model, roll_b, device, "thinning", carried)
            r["sim_thinning_ev_s"], r["sim_thinning_events"] = evs, n
        r["carried"] = carried
        out[tag] = r
        print(tag, json.dumps(r), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
