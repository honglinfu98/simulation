# Notation reference (carried over from the TFOW paper)

Source: *TFOW: Detecting Anomalous Order Flow in Crypto Limit Order Book* (pre-SS2P2 draft,
`reference/TFOW-5.pdf`). This is the notation system to keep consistent in the next paper.
Overload warnings at the bottom are things to fix, not keep.

## Events and marks

| Symbol | Meaning |
|---|---|
| $e = (c, s, \delta)$ | atomic event type: class, side, level |
| $c \in \{\mathrm{LO}, \mathrm{MO}, \mathrm{CO}, \mathrm{IS}\}$ | event class: limit order, market order, cancel order, inside-spread order |
| $s \in \{b, a\}$ | side (bid / ask) |
| $\delta \in \{1,\dots,K\}$ | level index; distance from best quote for LO/CO ($\delta{=}1$ = best); $\delta \equiv 1$ for MO; ticks of spread improvement for IS (e.g. $\mathrm{IS}_{b,2}$) |
| $K = 10$ | top-of-book depth kept per side |
| $\mathcal{E}$ (also $E$) | finite set of atomic event types |
| $M$ | number of atomic event types, $X_i \subseteq \{1,\dots,M\}$ |
| $X_{t_i} = \{(e_{i1}, v(e_{i1})), \dots, (e_{in_i}, v(e_{in_i}))\}$ | event **set** at time $t_i$ (simultaneous atomic events) |
| $v(e) \in \mathbb{R}_{>0}$, $\mathbf{v}_{t_i}$ | volume mark of atomic event $e$; the set of volumes at $t_i$ |
| $x_i = (e_i, v_i)$ | single event tuple (type, volume) |
| $\mathrm{cap}_t(e)$ | market-microstructure volume cap for event $e$ at time $t$ |

## Times, history, state

| Symbol | Meaning |
|---|---|
| $t_i$ (also $\tau_i$) | $i$-th event time; $t_1 < \dots < t_N \le T$ |
| $N$, $T$ | number of events; observation horizon |
| $H(t)$ / $\mathcal{H}_t$ | history of event sets strictly before $t$: $\{(\tau_j, X_j) : \tau_j < t\}$ |
| $\mathcal{F}_t$ | filtration (used in query/localization sections) |
| $h_t$ | continuous-time hidden state (decays between events, jumps at events) |
| $v_b^{(1)}(t), v_a^{(1)}(t)$ | top-of-book bid/ask queue volumes |
| $I(t) = \frac{v_b^{(1)} - v_a^{(1)}}{v_b^{(1)} + v_a^{(1)}}$ | queue-imbalance indicator |
| $s_t \in \{0, 1, 2\}$ | discrete market state (ask-dominant / balanced / bid-dominant), threshold $\theta \in (0,1)$ |
| $\phi \in \mathbb{R}^{K \times W \times W}$ | event-conditioned state-transition tensor (fit offline by counting); $W$ = number of states |

## Intensities and likelihood

| Symbol | Meaning |
|---|---|
| $\lambda^*(t)$ | ground (total) conditional intensity; $*$ = conditioning on history |
| $p^*(X \mid t, H(t))$ | conditional distribution over event sets |
| $\lambda^*_X(t) = \lambda^*(t)\, p^*(X \mid t, H(t))$ | marked (set) intensity factorization |
| $\lambda_k(t \mid H_t, s_t) = f_k(h_t, s_t)$ | per-type intensity decoded from latent + state; $f_\theta$ positive decoder (softplus) |
| $\lambda^{\mathrm{base}}_\theta$, $\lambda_0(t)$ | baseline (structural) intensity |
| $\Lambda^*(t) = \int_0^t \lambda^*(u)\,du$ | compensator; per-subset $\Lambda^*_A$, per-type $\Lambda^*_x$ |
| $u_i = \Lambda^*(t_i)$, $\Delta u_i = u_i - u_{i-1}$ | rescaled event times and compensator increments; $\Delta u_i \overset{iid}{\sim} \mathrm{Exp}(1)$ under correct specification |
| $\mathcal{L} = \mathcal{L}_{\mathrm{time}} + \mathcal{L}_{\mathrm{set}} (+\, \mathcal{L}_{\mathrm{volumes}})$ | log-likelihood decomposition of the set-MTPP |
| $\rho_k(h_t) = \sigma(w_k^\top n(h_t) + b_k)$ | Bernoulli set-head probability for type $k$ |
| $\mu_k(h_t), \sigma_k(h_t)$ | volume-head parameters (truncated log-normal) |

## Diagnostics / anomaly statistics (TFOW-specific)

| Symbol | Meaning |
|---|---|
| $\psi(Z) = \frac{1}{\Lambda^*(T)} \sum_{i=1}^{N+1} (u_i - u_{i-1})^2$ | sum-of-squared-spacings (3S) statistic |
| $r^{(2)}_i = (\Delta u_i)^2 - 2$ | differential 3S residual (centered second moment) |
| $d^{(2)}_i = \rho\, d^{(2)}_{i-1} + (1-\rho)\, r^{(2)}_i$ | EMA-smoothed deviation state, $\rho \in [0,1)$ |
| $\beta$ | learnable deviation-feedback gain: $\lambda = \lambda^{\mathrm{base}} \exp(\beta d^{(2)}_i)$ |
| $\rho^*(t) = \sum_k \lambda^*_k(t)\, \mathbb{E}[v \mid k, t, H_t]$, $\Xi^*(t)$ | mass (volume-flow) intensity and its compensator; $\psi_{\mathrm{mass}}$ analog of 3S |
| $\mathrm{hit}(A)$ | first time any event in subset $A$ occurs |
| $q(A \prec B) = \Pr(t_A < t_B)$ | A-before-B motif probability |
| $\lambda^*_S(t) = \lambda^*(t)\, p^*(S \mid t, H(t))$ | subset intensity for queries |
| $\alpha, \alpha_1, \alpha_2$ | detection / escalation significance thresholds |

## Overloads to FIX in the next paper (do not carry over)

- **$\rho$ is used three ways**: EMA smoothing factor, Bernoulli set-head probability $\rho_k$, and
  mass intensity $\rho^*(t)$. Suggest: keep $\rho_k$ for the set head; use $\gamma$ for the EMA factor
  and $m^*(t)$ (or $\varrho$) for mass intensity. (In the SS2P2 paper $\rho$ is also the branching
  ratio — reserve $\rho$ for the branching ratio and rename the rest.)
- **$s$ is used two ways**: order side $s \in \{b,a\}$ and market state $s_t$ (one section also calls the
  state $x_t$, colliding with the event tuple $x_i$). Suggest: side stays $s$; state becomes $z_t$;
  never $x_t$.
- **$K$ is used three ways**: book depth ($K{=}10$), number of event types (set head, $\phi$ tensor),
  and generic type count. Suggest: depth $L$ (levels), event-type count $K$ — matches the SS2P2
  paper where $K$ = mark count.
- **$\theta$ is used two ways**: imbalance threshold and model parameters $\lambda_\theta$. Suggest:
  threshold $\vartheta$ or $\eta$; parameters stay $\theta$.
- $\lambda^{\mathrm{base}}_\theta$ vs $\lambda_0$ and $H(t)$ vs $\mathcal{H}_t$ vs $\mathcal{F}_t$ are
  duplicated notations for the same objects across sections — pick one form each.

## Conventions worth keeping

- $*$ superscript for history-conditioning ($\lambda^*, \Lambda^*, p^*$).
- Time-rescaling language: $u_i$, $\Delta u_i \sim \mathrm{Exp}(1)$, "compensator clock".
- Set-MTPP factorization $\lambda^*_X = \lambda^* \, p^*(X \mid t, H)$ and
  $\mathcal{L}_{\mathrm{time}} + \mathcal{L}_{\mathrm{set}}$ split — this maps directly onto SS2P2's
  ground-intensity × rate-neutral mark head design.
- Atomic event grammar $(c, s, \delta)$ with classes LO/MO/CO/IS and level semantics.
