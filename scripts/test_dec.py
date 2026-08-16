"""Correctness tests for DEC-S2P2 (bounded rate x tower composition, disjoint).

The decomposition claim is the whole point of the model, so it is asserted here
by autograd rather than argued: the mark block must not be able to move Lambda,
the rate block must not be able to move p, and the two must share no tensors.
Test 5 checks the consequence that matters operationally -- with disjoint blocks
the time/mark loss weighting cannot change the rate solution, which is exactly
what failed in pct-s2p2 (E[u] 1.43 -> 2.76 while val loss fell monotonically).
"""
import sys, torch
sys.path.insert(0, '/Users/Ryan/simulation')
import torch.nn.functional as F
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp

torch.manual_seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {DEV}')
K, B, N = 12, 3, 32
CFG = dict(decoder_type='dec-s2p2', recurrent_hidden_size=16, channel_embedding_size=16,
           time_embedding_size=16, num_channels=K, target_rate=20.0, use_volume=False,
           s2p2_layers=2, pct_tower_dim=6, pct_state_dim=4, pct_layers=2)

def build(**kw):
    cfg = dict(CFG); cfg.update(kw)
    return create_volume_set_mtpp(K, cfg, DEV, use_volume=False).to(DEV).eval()

marks = torch.zeros(B, N, K); marks.scatter_(2, torch.randint(0, K, (B, N, 1)), 1.0)
marks = marks.to(DEV)
ts = torch.cumsum(torch.rand(B, N) * 0.1, dim=1).to(DEV)

ok = True
def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {detail}')

m = build(); d = m.decoder

print('1. the two blocks share no parameter tensors')
rp = {id(p) for p in d.rate.parameters()}
mp = {id(p) for p in d.mark_tower.parameters()}
check('no shared tensors', len(rp & mp) == 0, f'{len(rp & mp)} shared')
check('embeddings are distinct objects',
      d.rate.channel_embedding is not d.mark_tower.channel_embedding)
check('embedding weights are distinct tensors',
      id(d.rate.channel_embedding.weight) != id(d.mark_tower.channel_embedding.weight))

print('2. dLambda/d(mark params) == 0  and  dlogits/d(rate params) == 0')
right, left = d.get_states_and_event_left_states(marks, ts)
h = d.get_hidden_h(right, ts, ts - 1e-6)
lam = d.ground_intensity(h).sum()
mk = list(d.mark_tower.parameters())
g = torch.autograd.grad(lam, mk, retain_graph=True, allow_unused=True)
check('dLambda/d(theta_mark) == 0',
      all(x is None or float(x.abs().max()) == 0.0 for x in g))
z = d.mark_score(h).sum()
rk = list(d.rate.parameters())
g2 = torch.autograd.grad(z, rk, retain_graph=True, allow_unused=True)
check('dlogits/d(theta_rate) == 0',
      all(x is None or float(x.abs().max()) == 0.0 for x in g2))
# and each block DOES reach its own head
g3 = torch.autograd.grad(lam, [d.rate.rate_w.weight], retain_graph=True, allow_unused=True)[0]
check('dLambda/d(rate head) != 0', g3 is not None and float(g3.abs().max()) > 0)
g4 = torch.autograd.grad(z, [d.mark_tower.mark_bias], allow_unused=True)[0]
check('dlogits/d(mark head) != 0', g4 is not None and float(g4.abs().max()) > 0)

print('3. exact dominating rate survives the composition')
lo, hi = d.rate_bounds()
tot = d.ground_intensity(h)
check('Lambda <= ell_+', float(tot.max()) <= hi, f'{float(tot.max()):.3f} <= {hi:.3f}')
with torch.no_grad():
    d.rate.rate_w.bias.fill_(1e3)
check('cap binds under saturation',
      float(d.ground_intensity(d.get_hidden_h(right, ts, ts - 1e-6)).max()) <= hi + 1e-3)

print('4. carried state across a window split == one pass')
m = build(); d = m.decoder
full_r, full_l = d.get_states_and_event_left_states(marks, ts)
hh = N // 2
r1, _ = d.get_states_and_event_left_states(marks[:, :hh], ts[:, :hh])
r2, l2 = d.get_states_and_event_left_states(
    marks[:, hh:], ts[:, hh:] - ts[:, hh - 1:hh], old_states=r1[:, -1])
e = (l2 - full_l[:, hh:]).abs().max().item()
check('carried left-limits match', e < 1e-2, f'max|err| = {e:.3e}')

print('5. loss weighting cannot move the rate solution (the DTPP property)')
# Scale the mark term by 100x; rate gradients must be bit-identical.  Under
# pct-s2p2 this was false, and is why E[u] drifted with training.
mm = build(); dd = mm.decoder          # ONE model: build() advances the RNG, so
def rate_grads(mark_weight):           # two builds would differ by init, not by weight
    for prm in dd.parameters():
        prm.grad = None
    rr, ll = dd.get_states_and_event_left_states(marks, ts)
    hq = dd.get_hidden_h(rr, ts, ts - 1e-6)
    time_term = dd.ground_intensity(hq).sum()
    mark_term = (torch.log_softmax(dd.mark_score(hq), dim=-1) * marks).sum()
    (time_term + mark_weight * mark_term).backward()
    return [p.grad.clone() for p in dd.rate.parameters() if p.grad is not None]
g_a, g_b = rate_grads(1.0), rate_grads(100.0)
same = len(g_a) == len(g_b) and len(g_a) > 0 and all(torch.equal(a, b) for a, b in zip(g_a, g_b))
check('rate grads identical at 1x and 100x mark weight', same,
      f'({len(g_a)} tensors compared)')

print(f'\n{d.disjointness_report()}')
print('ALL PASS' if ok else 'FAILURES PRESENT')
sys.exit(0 if ok else 1)
