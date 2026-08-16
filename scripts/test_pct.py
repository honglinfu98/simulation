"""Correctness tests for PCT-S2P2 (CPU, no data needed).

Checks the things that are easy to get subtly wrong: the parallel scan against
a sequential loop, the left/right limit identity, the exactness of the thinning
ceiling, carried-state equivalence across window boundaries, and that the
mixer/impulse ablations actually sever the paths they claim to.
"""
import sys, math, torch
sys.path.insert(0, '/Users/Ryan/simulation')
import torch.nn.functional as F
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp

torch.manual_seed(0)
# run on GPU when present: a CPU-only self-test cannot see device-mismatch
# bugs (the block-diag mask hook closed over a CPU local -- job 7183266.3/6/9)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {DEV}')
K, B, N = 12, 3, 32
CFG = dict(decoder_type='pct-s2p2', recurrent_hidden_size=32, channel_embedding_size=16,
           time_embedding_size=16, num_channels=K, target_rate=20.0,
           pct_tower_dim=6, pct_state_dim=4, pct_layers=2, use_volume=False)

def build(**kw):
    cfg = dict(CFG); cfg.update(kw)
    return create_volume_set_mtpp(K, cfg, DEV, use_volume=False).to(DEV).eval()

def batch(n=N):
    marks = torch.zeros(B, n, K)
    marks.scatter_(2, torch.randint(0, K, (B, n, 1)), 1.0)
    ts = torch.cumsum(torch.rand(B, n) * 0.1, dim=1)
    return marks.to(DEV), ts.to(DEV)

ok = True
def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {detail}')

marks, ts = batch()

print('1. parallel scan == sequential recurrence')
from volume_set_mtpp.models.pct_s2p2_decoder import cscan
R = 5
a = torch.exp(torch.complex(-torch.rand(B, N, R), torch.randn(B, N, R)) * 0.1).to(DEV)
b = torch.complex(torch.randn(B, N, R), torch.randn(B, N, R)).to(DEV)
x0 = torch.complex(torch.randn(B, R), torch.randn(B, R)).to(DEV)
seq = torch.zeros(B, N, R, dtype=torch.cfloat, device=DEV); x = x0
for i in range(N):
    x = a[:, i] * x + b[:, i]; seq[:, i] = x
err = (cscan(x0, a, b) - seq).abs().max().item()
check('scan == loop', err < 1e-4, f'max|err| = {err:.3e}')

print('2. thinning ceiling is a valid upper bound on sum_k lambda_k')
m = build(); d = m.decoder
lo, hi = d.rate_bounds()
right, left = d.get_states_and_event_left_states(marks, ts)
lam = d.type_intensities(left)                       # [B,N,K]
tot = lam.sum(-1)
check('Lambda <= ell_+', float(tot.max()) <= hi, f'max {float(tot.max()):.3f} <= {hi:.3f}')
check('ell_+ closed form exact',
      abs(hi - float((F.softplus(d.raw_scale) * F.softplus(torch.tensor(d.c))).sum())) < 1e-4)
# saturate the head: huge z_raw must still respect the cap
with torch.no_grad():
    d.rate_w.bias.fill_(1e3)
lam_sat = d.type_intensities(left).sum(-1)
check('cap binds under saturation', float(lam_sat.max()) <= hi + 1e-3,
      f'saturated {float(lam_sat.max()):.3f} <= {hi:.3f}')
check('saturated == ceiling', abs(float(lam_sat.max()) - hi) / hi < 1e-3)

print('3. lambda > 0 everywhere (floor is exactly zero, not negative)')
m = build(); d = m.decoder
with torch.no_grad():
    d.rate_w.bias.fill_(-1e3)
lam_q = d.type_intensities(d.get_states_and_event_left_states(marks, ts)[1])
check('lambda_k > 0', float(lam_q.min()) > 0, f'min {float(lam_q.min()):.3e}')

print('4. carried state across a window split == one pass')
m = build(); d = m.decoder
full_right, full_left = d.get_states_and_event_left_states(marks, ts)
h = N // 2
r1, l1 = d.get_states_and_event_left_states(marks[:, :h], ts[:, :h])
carry = r1[:, -1]
r2, l2 = d.get_states_and_event_left_states(
    marks[:, h:], ts[:, h:] - ts[:, h - 1:h], old_states=carry)
e = (l2 - full_left[:, h:]).abs().max().item()
check('carried left-limits match', e < 1e-2, f'max|err| = {e:.3e}')

print('5. get_hidden_h -> t_i^- converges to the left limit')
# querying AT t_i with right=True returns the POST-event state; the left limit
# is the t -> t_i^- limit, so approach it from below.
m = build(); d = m.decoder
right, left = d.get_states_and_event_left_states(marks, ts)
prev = [ (d.get_hidden_h(right, ts, ts - eps) - left).abs().max().item()
         for eps in (1e-3, 1e-5, 1e-7) ]
check('h(t_i^-) -> left limit', prev[-1] < 1e-2, f'err @eps=1e-7: {prev[-1]:.3e}')
check('error shrinks with eps', prev[0] > prev[-1], f'{prev[0]:.2e} -> {prev[-1]:.2e}')
# and the post-event query must differ from the left limit by exactly the jump
hR = d.get_hidden_h(right, ts, ts)
check('h(t_i) != left limit (jump present)', (hR - left).abs().max() > 1e-4)

print('6. ablations sever the paths they claim to')
m_all = build(pct_impulse_mode='all'); m_own = build(pct_impulse_mode='own')
d_all, d_own = m_all.decoder, m_own.decoder
# impulse_mode='own': tower k must not move when a non-k event fires.  Compare
# layer-0 states (before any mixing) for a single event of type 0.
mk = torch.zeros(1, 1, K, device=DEV); mk[0, 0, 0] = 1.0
tt = torch.tensor([[0.5]])
def tower0_state(dec):
    emb = dec._impulse_emb(mk)                                   # [1,1,K,H]
    return emb[0, 0]                                             # [K,H]
e_own, e_all = tower0_state(d_own), tower0_state(d_all)
check("own: non-k towers get zero impulse", float(e_own[1:].abs().max()) == 0.0)
check("own: own tower gets nonzero impulse", float(e_own[0].abs().max()) > 0)
check("all: every tower gets the impulse", float(e_all[1:].abs().min()) > 0)

m_bd = build(pct_block_diag_mixers=True); W = m_bd.decoder.mixers[0].weight
Kk, Hh = m_bd.decoder.K, m_bd.decoder.H
Wb = W.reshape(Kk, Hh, Kk, Hh)
offdiag = sum(float(Wb[i, :, j, :].abs().max()) for i in range(Kk) for j in range(Kk) if i != j)
check('block-diag mixers: off-blocks zero', offdiag == 0.0, f'sum|offdiag| = {offdiag:.3e}')
check('block-diag mixers: diag blocks nonzero',
      min(float(Wb[i, :, i, :].abs().max()) for i in range(Kk)) > 0)

print('6b. block-diag mask survives a BACKWARD pass on the live device')
m = build(pct_block_diag_mixers=True); d = m.decoder
r, l = d.get_states_and_event_left_states(marks, ts)
d.type_intensities(l).sum().backward()
g = d.mixers[0].weight.grad.reshape(d.K, d.H, d.K, d.H)
off = sum(float(g[i, :, j, :].abs().max()) for i in range(d.K) for j in range(d.K) if i != j)
check('off-block GRADIENT zero', off == 0.0, f'sum = {off:.3e}')
check('diag-block gradient nonzero',
      max(float(g[i, :, i, :].abs().max()) for i in range(d.K)) > 0)

print('7. L=1 refused (towers would never mix)')
try:
    build(pct_layers=1); check('L=1 raises', False)
except ValueError:
    check('L=1 raises', True)

print('8. gradients flow to every block')
m = build(); d = m.decoder
right, left = d.get_states_and_event_left_states(marks, ts)
loss = d.type_intensities(left).sum()
loss.backward()
for name in ('alpha', 'raw_scale'):
    g = getattr(d, name).grad
    check(f'grad {name}', g is not None and float(g.abs().max()) > 0)
for nm, p in (('mixer0', d.mixers[0].weight), ('lam0', d.lam_lnr[0]), ('E0', d.E_re[0])):
    check(f'grad {nm}', p.grad is not None and float(p.grad.abs().max()) > 0)

print('9. MEX-S2P2: explicit K x K transition + per-tower decoupled heads')
mx = build(pct_impulse_mode='matrix', pct_per_tower_head=True); dx = mx.decoder
T = dx.transition_matrix()
check('transition_matrix is K x K', tuple(T.shape) == (dx.K, dx.K), f'{tuple(T.shape)}')
# column-L1 normalisation rescales each column to budget c_j, so the diagonal
# is no longer exactly 1; the meaningful init property is diagonal DOMINANCE
# (the model starts at "own" and must learn the off-diagonal).
_dg = T.diag().abs().mean(); _off = (T - torch.diag(T.diag())).abs().mean()
check('T init diagonally dominant', float(_dg / _off) > 20,
      f'diag/off = {float(_dg/_off):.1f}')
# a type-j event must reach EVERY tower through T (that is the mutual excitation)
mk1 = torch.zeros(1, 1, dx.K, device=DEV); mk1[0, 0, 0] = 1.0
eps = dx._impulse_emb(mk1)[0, 0]                       # [K,H]
check('type-0 event reaches all towers', float(eps.abs().min()) > 0,
      f'min |eps| = {float(eps.abs().min()):.3e}')
r9, l9 = dx.get_states_and_event_left_states(marks, ts)
lam9 = dx.type_intensities(l9)
lo9, hi9 = dx.rate_bounds()
check('bound still exact with per-tower heads', float(lam9.sum(-1).max()) <= hi9,
      f'{float(lam9.sum(-1).max()):.2f} <= {hi9:.2f}')
lam9.sum().backward()
check('grad reaches T', dx.trans_raw.grad is not None and float(dx.trans_raw.grad.abs().max()) > 0)
check('per-tower head params exist and get grad',
      dx.head_w.grad is not None and float(dx.head_w.grad.abs().max()) > 0)
# heads are independent: perturbing tower 0's head must not move lambda_1
with torch.no_grad():
    dx.head_b[0] += 5.0
lam_p = dx.type_intensities(l9)
d0 = float((lam_p[..., 0] - lam9[..., 0]).abs().max())
d1 = float((lam_p[..., 1] - lam9[..., 1]).abs().max())
check('tower heads independent', d0 > 0 and d1 == 0.0, f'd(lam_0)={d0:.3e}  d(lam_1)={d1:.1e}')
print('10. spectral control on the K x K routing (eigendecomposition)')
import numpy as np
sp = build(pct_impulse_mode='matrix', pct_per_tower_head=True,
           pct_c_max=2.0, pct_c_init=1.0).decoder
T = sp._T()
cb = sp.column_budgets()
check('column budgets == c_j exactly', float((cb - cb.mean()).abs().max()) < 1e-5,
      f'spread = {float((cb-cb.mean()).abs().max()):.2e}')
rho = sp.spectral_radius()
ev = np.linalg.eigvals(T.detach().cpu().numpy())
check('rho matches numpy eigvals', abs(rho - float(np.abs(ev).max())) < 1e-4,
      f'{rho:.6f} vs {float(np.abs(ev).max()):.6f}')
check('rho <= ||T||_1 (the cheap bound)', rho <= float(T.abs().sum(0).max()) + 1e-5,
      f'{rho:.4f} <= {float(T.abs().sum(0).max()):.4f}')
# projection must enforce the bound and PRESERVE structure (sign pattern + ratios)
with torch.no_grad():
    sp.trans_mag.fill_(3.0)                      # push c_j near c_max
T_pre = sp._T().detach().clone(); rho_pre = sp.spectral_radius()
sp.project_spectral(0.5)
T_post = sp._T().detach(); rho_post = sp.spectral_radius()
check('projection enforces rho <= rho_max', rho_post <= 0.5 + 1e-3,
      f'{rho_pre:.4f} -> {rho_post:.4f}')
r = (T_post / T_pre.clamp(min=1e-12).abs().clamp(min=1e-12) * T_pre.abs().clamp(min=1e-12) / T_pre.clamp(min=1e-12).abs())
check('projection preserves signs', bool((torch.sign(T_post) == torch.sign(T_pre)).all()))
ratio = (T_post / T_pre).flatten()
check('projection is a uniform rescale', float(ratio.std()) < 1e-4,
      f'std(T_post/T_pre) = {float(ratio.std()):.2e}')
# differentiable penalty
sp2 = build(pct_impulse_mode='matrix', pct_per_tower_head=True).decoder
pen = sp2.spectral_penalty(0.1)
check('penalty > 0 when sigma_max > rho_max', float(pen) > 0, f'pen = {float(pen):.4f}')
pen.backward()
check('penalty grad reaches trans_raw',
      sp2.trans_raw.grad is not None and float(sp2.trans_raw.grad.abs().max()) > 0)
check('penalty grad reaches trans_mag',
      sp2.trans_mag.grad is not None and float(sp2.trans_mag.grad.abs().max()) > 0)
check('penalty == 0 when rho_max is large', float(sp2.spectral_penalty(1e6)) == 0.0)

print('ALL PASS' if ok else 'FAILURES PRESENT')

n_par = sum(p.numel() for p in m.parameters())
print(f'\nK={K} H={d.H} P={d.P} L={d.L}  params={n_par:,}  ell_+={d.rate_bounds()[1]:.2f} ev/s')
sys.exit(0 if ok else 1)

