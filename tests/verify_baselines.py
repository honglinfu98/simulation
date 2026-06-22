"""Verification for the four new baseline decoders (lstm, sahp, ct-lstm, pct-lstm).

Builds the full model via create_volume_set_mtpp with a tiny categorical config,
runs forward + compute_loss + backward on synthetic data, and checks anti-leakage
for the two generic decoders.

    PYTHONPATH=. python3 tests/verify_baselines.py
"""
import sys
import torch

from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp

K, B, N = 62, 4, 20


def tiny_config(decoder_type):
    return {
        'channel_embedding_size': 32,
        'time_embedding_size': 16,
        'recurrent_hidden_size': 64,
        'decoder_type': decoder_type,
        'mark_head': 'categorical',
        'sahp_heads': 4,
        'sahp_layers': 2,
    }


def synth():
    torch.manual_seed(0)
    marks = torch.zeros(B, N, K)
    for b in range(B):
        for i in range(N):
            if torch.rand(1).item() > 0.2:
                marks[b, i, torch.randint(0, K, (1,)).item()] = 1.0
    dt = torch.rand(B, N) * 0.3 + 1e-3
    tgt_dt = torch.rand(B) * 0.3 + 1e-3
    tgt_marks = torch.zeros(B, K)
    for b in range(B):
        tgt_marks[b, torch.randint(0, K, (1,)).item()] = 1.0
    return {
        'input_times': dt,
        'input_marks': marks,
        'target_time': tgt_dt,
        'target_marks': tgt_marks,
    }


def check_anti_leakage(model, batch):
    """left[:,0] must be the init/zero state, independent of event 0's mark."""
    ts = torch.cumsum(batch['input_times'], dim=1)
    _, left = model.decoder.get_states_and_event_left_states(batch['input_marks'], ts)
    zero_ok = torch.allclose(left[:, 0], torch.zeros_like(left[:, 0]))
    # Independence: perturb event-0 mark, left[:,0] must not change.
    m2 = batch['input_marks'].clone()
    m2[:, 0] = 0.0
    m2[:, 0, (m2[:, 0].argmax(-1) + 1) % K] = 1.0
    _, left2 = model.decoder.get_states_and_event_left_states(m2, ts)
    indep_ok = torch.allclose(left[:, 0], left2[:, 0])
    return zero_ok, indep_ok


def main():
    device = torch.device('cpu')
    fails = 0
    for dt in ['lstm', 'sahp', 'ct-lstm', 'pct-lstm']:
        try:
            model = create_volume_set_mtpp(num_channels=K, config=tiny_config(dt), device=device,
                                           use_volume=False, intensity_type='dynamic')
            model.train()
            batch = synth()
            loss, metrics = model.compute_loss(batch, device)
            model.zero_grad()
            loss.backward()
            n_grad = sum(1 for p in model.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
            finite = torch.isfinite(loss).item()
            grad_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

            al = ""
            if dt in ('lstm', 'sahp'):
                zero_ok, indep_ok = check_anti_leakage(model, batch)
                al = f" | anti-leak: left[:,0]==0 {zero_ok}, indep-of-ev0 {indep_ok}"
                if not (zero_ok and indep_ok):
                    fails += 1
            ok = finite and grad_finite and n_grad > 0
            print(f"  {'PASS' if ok else 'FAIL'}  {dt:9s} | loss={loss.item():.4f} finite={finite} "
                  f"grad_finite={grad_finite} n_grad={n_grad}{al}")
            if not ok:
                fails += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {dt}: {e}")
            fails += 1
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
