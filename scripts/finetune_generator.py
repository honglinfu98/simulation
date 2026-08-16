#!/usr/bin/env python3
"""Fit the mode-space generator on top of a trained LGM checkpoint.

The generator adds R latent modes with a dense generator G, parameterized in
G's eigenbasis (lam_r = -delta_r + i omega_r, P folded into the emission /
readout maps).  It contributes ONLY to the mark logits, which are softmax
normalized, so:

  * the total intensity never sees it  -> n, the mu_0 pin and the Fano factor
    are invariant BY CONSTRUCTION, not by tuning.  scripts/test_gen.py asserts
    d(Lambda)/d(gen params) == 0 by autograd;
  * the log-likelihood stays additively separable -> the transplant theorem
    still holds, so this checkpoint can still donate/receive a ground.

Hence this finetune freezes everything except gen_*: the backbone, the mark
head and the ground are bit-identical to the donor afterwards, and the ONLY
thing that can move is the mark composition.  With gen_U zero-init the model
starts exactly at the donor, so epoch 0 reproduces the donor's metrics.

Because the ground is frozen and the time-likelihood has no gradient path to
the generator, the optimizer sees only the mark term; the full loss is used
anyway so the reported numbers stay comparable with the training runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import torch

from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    tot, n, tot_time = 0.0, 0, 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            _loss, m = model.compute_loss(move_batch(batch, device), device)
            tot += m['set_nll']
            tot_time += m['time_nll']
            n += 1
    return (tot / max(n, 1)), (tot_time / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='donor LGM checkpoint (e.g. lgm-kf2-sN)')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--tau-min', type=float, default=0.05)
    ap.add_argument('--tau-max', type=float, default=120.0)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--seq-length', type=int, default=4096)
    ap.add_argument('--stride', type=int, default=4096)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-files', type=int, default=7)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--val-batches', type=int, default=40)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(ck['config'])
    assert cfg.get('decoder_type') == 'lgm', cfg.get('decoder_type')
    assert int(cfg.get('lgm_gen_rank', 0)) == 0, 'donor already has a generator'
    cfg['lgm_gen_rank'] = args.rank
    cfg['lgm_gen_tau_min'] = args.tau_min
    cfg['lgm_gen_tau_max'] = args.tau_max
    K = cfg.get('num_channels', 62)

    model = create_volume_set_mtpp(K, cfg, device, use_volume=cfg.get('use_volume', False))
    missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
    # the ONLY thing the donor may be missing is the generator block
    assert not unexpected, f'unexpected donor keys: {unexpected}'
    assert all('gen_' in k for k in missing), f'non-generator keys missing: {missing}'
    print(f'LOADED donor; fresh params: {sorted(missing)}', flush=True)
    model.to(device)

    gen_params, frozen = [], 0
    for name, p in model.named_parameters():
        if 'gen_' in name:
            p.requires_grad_(True)
            gen_params.append(p)
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    n_train = sum(p.numel() for p in gen_params)
    print(f'TRAINABLE {n_train} generator params; FROZEN {frozen} '
          f'({100.0 * n_train / (n_train + frozen):.3f}% trainable)', flush=True)

    # fingerprint the frozen ground: re-asserted at the end, so "the ground did
    # not move" is verified rather than assumed.
    d = model.decoder
    rho0 = d.closed_form_rho()
    ground0 = {k: v.detach().clone() for k, v in d.state_dict().items()
               if k in ('a_raw', 'log_delta_g') or k.startswith('mark.')}

    train_loader, val_loader, _test_loader, _em = create_bfnx_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        sequence_length=args.seq_length, stride=args.stride,
        max_files=args.max_files, cache_dir=args.cache_dir)

    base_set, base_time = evaluate(model, val_loader, device, args.val_batches)
    print(f'EPOCH 0 (donor) val set_nll={base_set:.6f} time_nll={base_time:.6f}', flush=True)
    print(f'  {d.gen_summary()}', flush=True)

    opt = torch.optim.AdamW(gen_params, lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best = base_set
    best_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items() if 'gen_' in k}
    history = [{'epoch': 0, 'val_set_nll': base_set, 'val_time_nll': base_time}]

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for batch in train_loader:
            loss, m = model.compute_loss(move_batch(batch, device), device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt.step()
            sched.step()
            run += m['set_nll']
            nb += 1
            if nb % 50 == 0:
                print(f'  ep{ep} step {nb}/{len(train_loader)} '
                      f'train set_nll={run / nb:.6f}', flush=True)
        vs, vt = evaluate(model, val_loader, device, args.val_batches)
        print(f'EPOCH {ep} train set_nll={run / max(nb, 1):.6f} '
              f'val set_nll={vs:.6f} time_nll={vt:.6f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        print(f'  {d.gen_summary()}', flush=True)
        history.append({'epoch': ep, 'train_set_nll': run / max(nb, 1),
                        'val_set_nll': vs, 'val_time_nll': vt})
        if vs < best:
            best = vs
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items() if 'gen_' in k}
            print(f'  new best val set_nll={best:.6f}', flush=True)

    model.load_state_dict(best_state, strict=False)

    # ---- verify the invariants the design rests on -------------------------
    rho1 = d.closed_form_rho()
    assert abs(rho1 - rho0) < 1e-9, f'branching moved: {rho0} -> {rho1}'
    for k, v0 in ground0.items():
        v1 = d.state_dict()[k]
        assert torch.equal(v0.cpu(), v1.cpu()), f'frozen tensor {k} moved'
    print(f'VERIFIED ground + mark head unchanged; n={rho1:.6f} '
          f'(donor {rho0:.6f})', flush=True)

    ck_out = dict(ck)
    ck_out['config'] = cfg
    sd = model.state_dict()
    ck_out['model_state_dict'] = {k: v.detach().cpu() for k, v in sd.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck_out, args.out)

    M = d.gen_transition_matrix().cpu()
    delta, omega = d._gen_deltas().detach().cpu(), d.gen_omega.detach().cpu()
    side = json.dumps({
        'donor': args.checkpoint, 'rank': args.rank,
        'val_set_nll_donor': base_set, 'val_set_nll_best': best,
        'improvement_nats': base_set - best,
        'val_time_nll_donor': base_time,
        'branching_n': rho1,
        'mode_timescales_s': (1.0 / delta).tolist(),
        'mode_periods_s': [float('inf') if abs(w) < 1e-3 else 2 * math.pi / abs(float(w))
                           for w in omega],
        'transition_matrix': M.tolist(),
        'history': history,
    }, indent=1)
    with open(os.path.splitext(args.out)[0] + '_gen.json', 'w') as f:
        f.write(side)
    print(f'WROTE {args.out}', flush=True)
    print(f'FINAL donor_set_nll={base_set:.6f} best_set_nll={best:.6f} '
          f'delta={base_set - best:+.6f} nats', flush=True)


if __name__ == '__main__':
    main()
