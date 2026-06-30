# Theory

This page sets out the mathematics behind each module. Notation: $E$ is exposure
(member-months), $C$ total incurred claims, $Z$ credibility, $t$ an annual trend
rate, and $\Delta$ the time in years between the midpoint of the experience
period and the midpoint of the rating period.

## Credibility

### Limited fluctuation (classical)

The full-credibility standard is the expected number of claims for the observed
aggregate to fall within $\pm k$ of its mean with probability $p$:

$$
n_F = \left(\frac{z_{(1+p)/2}}{k}\right)^2 \left(1 + \mathrm{CV}_S^2\right),
$$

where $z_{(1+p)/2}=\Phi^{-1}\!\big((1+p)/2\big)$ and the severity term
$\mathrm{CV}_S^2$ is dropped for a pure-frequency (Poisson) standard. For
$p=0.90,\,k=0.05$ this gives $n_F=(1.645/0.05)^2\approx1082$ claims.

Partial credibility follows the **square-root rule**:

$$
Z = \min\!\left(1,\ \sqrt{n / n_F}\right).
$$

### Bühlmann (greatest accuracy)

$$
Z = \frac{n}{n+k}, \qquad k = \frac{\text{EPV}}{\text{VHM}}
= \frac{\mathbb{E}\big[\mathrm{Var}(X\mid\Theta)\big]}{\mathrm{Var}\big(\mathbb{E}[X\mid\Theta]\big)}.
$$

EPV is the expected process variance; VHM is the variance of the hypothetical
means. A risk is more credible when between-risk variation (VHM) is large
relative to within-risk noise (EPV).

### Bühlmann-Straub (empirical, with exposures)

For groups $i=1,\dots,r$ with exposures $m_{ij}$ and per-unit observations
$X_{ij}$, let $m_i=\sum_j m_{ij}$, $\bar X_i=\sum_j m_{ij}X_{ij}/m_i$,
$m=\sum_i m_i$, and $\bar X=\sum_i m_i\bar X_i/m$. The structural parameters are
estimated by

$$
\hat s^2 = \frac{\sum_i\sum_j m_{ij}\,(X_{ij}-\bar X_i)^2}{\sum_i (n_i-1)},
\qquad
\hat a = \frac{\sum_i m_i(\bar X_i-\bar X)^2 - (r-1)\hat s^2}
              {m - \sum_i m_i^2 / m},
$$

with $k=\hat s^2/\hat a$ and $Z_i = m_i/(m_i+k)$. A negative $\hat a$ is
truncated to zero (no credibility). The credibility-weighted estimate for group
$i$ is $Z_i\bar X_i + (1-Z_i)\bar X$.

## Trend

The trend factor compounds the annual rate over the midpoint gap:

$$
\text{factor} = (1+t)^{\Delta}, \qquad
\Delta = \frac{m_{\text{rate}} - m_{\text{exp}}}{365.25}.
$$

A total trend splits multiplicatively into utilization and unit-cost
components: $(1+t) = (1+t_u)(1+t_c)$.

## Manual rate

$$
\text{manual claims PMPM} = \text{base} \times \prod_i f_i,
\qquad
\text{manual rate} = \frac{\text{manual claims PMPM}}{\text{target LR}},
$$

where the $f_i$ are rating relativities (area, industry, group size, plan/benefit,
network, ...) and member-level demographic factors are first aggregated to a
single membership-weighted relativity.

## Experience rate

Large claims are pooled at a point $P$, removing the excess
$\text{excess}=\sum_i\max(0, c_i-P)$. The developed experience claims PMPM is

$$
\text{exp claims PMPM}
  = \frac{C - \text{excess}}{E}\,(1+t)^{\Delta}\, f_{\text{ben}} f_{\text{demo}}
  + \text{pooling charge},
$$

then loaded to a charged rate by dividing by the target loss ratio.

## Blending and indication

Credibility blends experience and manual **claims**:

$$
\text{blended claims} = Z\,(\text{exp claims}) + (1-Z)\,(\text{man claims}).
$$

**Build-up method.** Gross up the blended claims and compare to the current rate:

$$
\text{indicated rate} = \frac{\text{blended claims}}{\text{target LR}},
\qquad
\text{change} = \frac{\text{indicated rate}}{\text{current rate}} - 1.
$$

**Loss-ratio method.** Weight the experience indication against a trend-only
("no experience") indication:

$$
\text{change} = Z\!\left(\frac{\text{exp LR}}{\text{target LR}} - 1\right)
              + (1-Z)\,\big((1+t)^{\Delta}-1\big),
\qquad
\text{exp LR} = \frac{\text{experience claims}}{\text{on-level premium}}.
$$

## Rate-change decomposition

A total change factor $F=\text{indicated}/\text{current}$ is written as a product
of driver factors. **Multiplicative:** $F=\prod_i f_i$. **Additive (percentage
points)**, by log-share normalization so the parts sum exactly to the total:

$$
c_i = \frac{\ln f_i}{\ln F}\,(F-1), \qquad \sum_i c_i = F - 1.
$$

When supplied factors do not multiply to an independently computed $F$, a
`residual` factor $F/\prod_i f_i$ is added so the attribution reconciles exactly
and the unexplained movement (e.g. rate adequacy / loading) is explicit.

## GLM relativities

A log-link GLM estimates relativities **jointly**, correcting for correlation
between rating variables that one-way ratios cannot. With
$\eta = X\beta + \text{offset}$ and $\text{offset}=\ln E$, the model is fit by
iteratively reweighted least squares. For a working response and weights

$$
z = \eta + (y-\mu)\,g'(\mu), \qquad w = \frac{w_{\text{prior}}}{V(\mu)\,g'(\mu)^2},
$$

the update is $\beta \leftarrow (X^\top W X)^{-1} X^\top W z$, iterated to
convergence. For the log link $g'(\mu)=1/\mu$, so $z=\eta+(y-\mu)/\mu$ and
$w=w_{\text{prior}}\,\mu^{2-p}$ under the variance function $V(\mu)=\mu^p$:

| Family | $p$ | Typical use |
| --- | --- | --- |
| Poisson | $1$ | claim frequency (counts, exposure offset) |
| Tweedie | $1<p<2$ | pure premium (mass at zero + continuous positive) |
| Gamma | $2$ | claim severity |

A level's relativity is $\exp(\hat\beta)$ relative to the base level, whose
relativity is $1$. The base level defaults to the most populous level (the most
stable choice) and can be set explicitly.

## Base rate and off-balance

The base rate is the cost level for the reference cell (all relativities 1). It
is backed out of the book so that base times relativities reproduces observed
losses. For risks $i$ with exposure $e_i$, relativity $r_i=\prod_k f_{ki}$, and
trended/developed loss $L_i$:

$$
\bar r = \frac{\sum_i e_i r_i}{\sum_i e_i}, \qquad
B = \frac{\sum_i L_i}{\sum_i e_i r_i} = \frac{\bar L}{\bar r},
$$

so $\sum_i e_i\,B r_i = \sum_i L_i$ by construction. When relativities are
revised, the exposure-weighted average moves from $\bar r_0$ to $\bar r_1$ and
the overall level drifts unless the base is **off-balanced**. To hold the level
neutral and then apply an intended overall change $\Delta$:

$$
B_1 = B_0\,\frac{\bar r_0}{\bar r_1}\,(1+\Delta),
$$

where $\bar r_0/\bar r_1$ is the off-balance correction.

## Retention and the gross-up

The charged (gross) rate follows the **fundamental insurance equation**. With
loss & LAE per member $L(1+\text{lae})$, flat fixed expense per member $F$,
variable load $V$ (percent-of-premium: commission, premium tax, fees,
%-admin), and profit/contingency $Q$ (percent of premium):

$$
P = L(1+\text{lae}) + F + V P + Q P
\;\Longrightarrow\;
P = \frac{L(1+\text{lae}) + F}{1 - V - Q}.
$$

$V$ appears in the denominator because premium tax and commission are levied on
the premium that already includes them. The permissible / target loss ratio is
then an **output**, not an input:

$$
\text{PLR} = \frac{L}{P} = \frac{L\,(1 - V - Q)}{L(1+\text{lae}) + F}.
$$

Because $F$ is added per member (after the base $\times$ relativity step), it
stays flat across rate cells, and total charged premium reconciles to
$\big(\sum_i L_i(1+\text{lae}) + F\sum_i e_i\big)/(1 - V - Q)$. With no fixed
expense the PLR collapses to the constant $1 - V - Q$.

## The rate build-up

A manual rate is assembled by an ordered sequence of operations on a running
total $v$, starting from a base claim cost:

| Operation | Effect |
| --- | --- |
| start | $v \leftarrow b$ |
| multiply | $v \leftarrow v\,f$ (a relativity or trend) |
| add | $v \leftarrow v + a$ (copay credit $a<0$; per-member fee $a>0$) |
| segment_multiply | $v \leftarrow v\,(1 - w + w f)$ (factor $f$ on a fraction $w$) |
| checkpoint | record a labeled subtotal; $v$ unchanged |

The evaluator records every step in a breakdown that reconciles row to row,
which is the artifact a reviewer or filing wants. Streams then combine: the
in-/out-of-network blend is $\text{par}\,p + \text{nonpar}\,(1-p)$, and medical
and drug costs add. The package supplies this grammar and the combine
operations; the factor *values* (cost-sharing, age/sex, area, ...) come from
filed tables and continuance/AV work and are taken as inputs.
