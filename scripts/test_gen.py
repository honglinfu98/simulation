"""Correctness tests for the mode-space generator (CPU, no data needed)."""
import sys, math, torch
sys.path.insert(0, '/Users/Ryan/simulation')
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp

torch.manual_seed(0)
K, B, N = 62, 3, 64
CFG = dict(decoder_type='lgm', recurrent_hidden_size=32, s2p2_layers=2,
           channel_embedding_size=16, time_embedding_size=16,
           num_channels=K, target_rate=20.0,
           lgm_timescales=6, use_volume=False)

def build(R):
    cfg = dict(CFG); cfg['lgm_gen_rank'] = R
    m = create_volume_set_mtpp(K, cfg, torch.device('cpu'), use_volume=False)
    return m.eval()

def batch():
    marks = torch.zeros(B, N, K)
    idx = torch.randint(0, K, (B, N))
    marks.scatter_(2, idx.unsqueeze(-1), 1.0)
    dts = torch.rand(B, N) * 0.1
    ts = torch.cumsum(dts, dim=1)
    return marks, ts

marks, ts = batch()
ok = True
def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {detail}')

# ---------------------------------------------------------------- 1. scan
print('1. complex scan vs sequential loop')
m32 = build(32); d = m32.decoder
delta, omega = d._gen_deltas(), d.gen_omega
lam = torch.complex(-delta, omega)
prev = torch.cat([torch.zeros_like(ts[:, :1]), ts[:, :-1]], 1)
dt = (ts - prev).clamp(min=0.0)
abar = torch.exp(lam[None, None] * dt.unsqueeze(-1))
e = torch.complex((marks @ d.gen_V), torch.zeros(B, N, 32))
Zl = torch.zeros(B, N, 32, dtype=torch.cfloat)
z = torch.zeros(B, 32, dtype=torch.cfloat)          # Z_right(-1) = 0
for i in range(N):
    z = abar[:, i] * z                               # Z_left(i)
    Zl[:, i] = z
    z = z + e[:, i]                                  # Z_right(i)
_, left_packed = d._gen_scan(ts, marks, None)
got = torch.complex(left_packed[..., :32], left_packed[..., 32:])
err = (got - Zl).abs().max().item()
check('scan == sequential', err < 1e-4, f'max|err| = {err:.3e}')

# ------------------------------------------------- 2. zero-init is identity
print('2. zero-init readout reproduces the R=0 model exactly')
m0 = build(0)
sd = m0.state_dict()
m32.load_state_dict({k: v for k, v in sd.items()}, strict=False)   # share backbone/head
for mm in (m0, m32):
    torch.manual_seed(1)
r0 = m0.decoder.get_states_and_event_left_states(marks, ts)
r32 = m32.decoder.get_states_and_event_left_states(marks, ts)
h0 = m0.decoder.get_hidden_h(r0[0][:, :-1], ts, ts)
h32 = m32.decoder.get_hidden_h(r32[0][:, :-1], ts, ts)
z0 = m0.decoder.mark_score(h0)
z32 = m32.decoder.mark_score(h32)
check('mark logits identical', (z0 - z32).abs().max() < 1e-6,
      f'max|d| = {(z0 - z32).abs().max():.3e}')
l0 = m0.decoder.ground_intensity(h0)
l32 = m32.decoder.ground_intensity(h32)
check('ground intensity identical', (l0 - l32).abs().max() < 1e-6,
      f'max|d| = {(l0 - l32).abs().max():.3e}')
check('closed-form n identical',
      abs(m0.decoder.closed_form_rho() - m32.decoder.closed_form_rho()) < 1e-9)

# --------------------------------------- 3. generator cannot touch the ground
print('3. generator params have ZERO gradient path to the total intensity')
m = build(16)
torch.nn.init.normal_(m.decoder.gen_U_re, std=0.5)
torch.nn.init.normal_(m.decoder.gen_U_im, std=0.5)
r = m.decoder.get_states_and_event_left_states(marks, ts)
h = m.decoder.get_hidden_h(r[0][:, :-1], ts, ts)
lam_tot = m.decoder.ground_intensity(h).sum()
g = torch.autograd.grad(lam_tot, [m.decoder.gen_U_re, m.decoder.gen_V],
                        retain_graph=True, allow_unused=True)
check('dLambda/dU == 0', g[0] is None or g[0].abs().max() == 0)
check('dLambda/dV == 0', g[1] is None or g[1].abs().max() == 0)
# and the marks DO see it
zz = m.decoder.mark_score(h).sum()
gm = torch.autograd.grad(zz, [m.decoder.gen_U_re], allow_unused=True)[0]
check('dlogits/dU != 0', gm is not None and gm.abs().max() > 0,
      f'max = {gm.abs().max():.3e}' if gm is not None else '')

# ------------------------------------------------ 4. carried rollout == batch
print('4. one-event-at-a-time carry reproduces the full-window scan')
m = build(16)
torch.nn.init.normal_(m.decoder.gen_U_re, std=0.3)
full_right, _ = m.decoder.get_states_and_event_left_states(marks, ts)
carry = full_right[:, 0]                       # state after 0 events (Z0, S0)
L, H = m.decoder.num_layers, m.decoder.recurrent_hidden_size
packed = None
for i in range(N):
    old = packed if packed is not None else None
    step_dt = (ts[:, i] - (ts[:, i - 1] if i > 0 else torch.zeros(B)))
    rt = m.decoder.get_states_and_event_left_states(
        marks[:, i:i + 1], step_dt.unsqueeze(1), old_states=old)[0]
    packed = rt[:, -1]
E = m.decoder._extra_dim
err_g = (packed[:, -E:-2 * 16] - full_right[:, -1, -E:-2 * 16]).abs().max().item()
err_z = (packed[:, -2 * 16:] - full_right[:, -1, -2 * 16:]).abs().max().item()
check('carried ground state matches', err_g < 1e-3, f'max|err| = {err_g:.3e}')
check('carried generator state matches', err_z < 1e-3, f'max|err| = {err_z:.3e}')

# -------------------------------------------------- 5. transition matrix form
print('5. induced transition matrix == numerical lag integral')
m = build(8)
torch.nn.init.normal_(m.decoder.gen_U_re, std=0.5)
torch.nn.init.normal_(m.decoder.gen_U_im, std=0.5)
M_closed = m.decoder.gen_transition_matrix()
d = m.decoder
delta, omega = d._gen_deltas(), d.gen_omega
s = d._gen_scale()
# integrate to ~25 slowest-mode timescales; midpoint rule
dtau = 0.005
taus = torch.arange(0.0, 3000.0, dtau) + 0.5 * dtau
# int_0^inf Re( U s exp(lam tau) ) V^T dtau, done numerically per mode
acc = torch.zeros(d.K, d.K)
for r in range(d.gen_R):
    amp = torch.exp(-delta[r] * taus)
    ic = float((amp * torch.cos(omega[r] * taus)).sum() * dtau)
    is_ = float((amp * torch.sin(omega[r] * taus)).sum() * dtau)
    ur, ui = d.gen_U_re[:, r] * s[r], d.gen_U_im[:, r] * s[r]
    acc += torch.outer(ur * ic - ui * is_, d.gen_V[:, r])
rel = (M_closed - acc).abs().max() / M_closed.abs().max()
check('closed form == numerical integral', rel < 1e-3, f'rel err = {rel:.3e}')

print(f'\n{d.gen_summary()}')
print('\nALL PASS' if ok else '\nFAILURES PRESENT')
sys.exit(0 if ok else 1)
