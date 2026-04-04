# Technical Report: DFL Regression Target Clamping Bug

## Problem Statement

The 0.5x NanoDet-Plus-Lite model systematically predicts bounding boxes
whose **width and height are smaller than the ground truth** for
medium-to-large objects. The issue is most visible on the 0.5x backbone
but affects all backbone sizes.

```
  Ground Truth vs Predicted (0.5x model, before fix)
  ┌──────────────────────────────────────────┐
  │                                          │
  │   ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐    │  ─ ─  GT box
  │   ╎  ┌───────────────────────────┐  ╎    │  ───  Predicted box
  │   ╎  │                           │  ╎    │
  │   ╎  │                           │  ╎    │  The predicted box is
  │   ╎  │      Object               │  ╎    │  systematically smaller
  │   ╎  │                           │  ╎    │  in both width and height.
  │   ╎  │                           │  ╎    │
  │   ╎  └───────────────────────────┘  ╎    │
  │   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘    │
  │                                          │
  └──────────────────────────────────────────┘
```

---

## Background: How NanoDet-Plus Regresses Bounding Boxes

NanoDet-Plus uses **Generalized Focal Loss v2** (GFLv2) for box
regression. Instead of directly predicting four scalar offsets, it
predicts a **discrete probability distribution** over a set of integer
distance bins and takes the **expected value** (integral) of that
distribution as the final offset.

### 1. Grid Priors

For each feature-map level with stride $s \in \{8, 16, 32, 64\}$, the
detector places a **center prior** at the top-left corner of every grid
cell:

$$
(p_x,\; p_y) = (j \cdot s,\; i \cdot s)
$$

where $i, j$ are the row/column indices of the feature map.

```
  Feature map (stride = 16, on a 320×320 input → 20×20 grid)

  0    16   32   48   64   80  ...  304
  ·────·────·────·────·────·── ... ──·   0
  │    │    │    │    │    │         │
  ·────·────·────·────·────·── ... ──·   16
  │    │    │    │    │    │         │
  ·────·────·────·────·────·── ... ──·   32
  │    │    │    │    │    │         │
  :    :    :    :    :    :         :
  ·────·────·────·────·────·── ... ──·   304

  · = center prior (placed at top-left of each cell)
  Each · predicts 4 distances: left, top, right, bottom
```

### 2. Distance Parameterisation

A bounding box $(x_1, y_1, x_2, y_2)$ is encoded as four
**distances** from the center prior to each edge:

$$
l = p_x - x_1,\quad
t = p_y - y_1,\quad
r = x_2 - p_x,\quad
b = y_2 - p_y
$$

```
              t (top)
         ┌──────┴──────┐
         │             │
  l ── · P(px,py)      │ ── r (right)
 (left)  │             │
         └──────┬──────┘
              b (bottom)

  Box = (px - l,  py - t,  px + r,  py + b)
```

All four distances are non-negative when the center prior lies inside the box.

### 3. Distribution Focal Loss (DFL) Representation

Each distance is **normalised** by the stride to give a unit-free target:

$$
\hat{d} = \frac{d}{s}, \qquad d \in \{l, t, r, b\}
$$

The model predicts a distribution over $R + 1$ discrete bins
$\{0, 1, \ldots, R\}$, where $R$ = `reg_max` (default 7).
The predicted distance is the **expected value** (integral):

$$
\tilde{d} = \sum_{k=0}^{R} k \cdot \text{softmax}(\mathbf{z})_k
$$

where $\mathbf{z} \in \mathbb{R}^{R+1}$ are the raw logits. This
output is bounded:

$$
\tilde{d} \in [0,\; R]
$$

```
  DFL Distribution Example (reg_max = 7, target d̂ = 4.3)

  The model learns a probability distribution over 8 bins [0..7].
  Expected value = sum of (bin × probability) = predicted distance.

  Probability
   0.5 │
       │         ██
   0.4 │         ██
       │      ██ ██
   0.3 │      ██ ██
       │      ██ ██
   0.2 │      ██ ██
       │      ██ ██ ██
   0.1 │   ██ ██ ██ ██
       │   ██ ██ ██ ██ ██
   0.0 └──██─██─██─██─██─██─██─██──
       0   1   2   3   4   5   6   7     ← bin index (distance units)
                      ↑
                   target = 4.3
                   (soft two-hot at bins 4 and 5)
```

The actual pixel-space distance is recovered by multiplying back by the
stride:

$$
d_{\text{pred}} = \tilde{d} \cdot s
$$

### 4. DFL Loss

DFL supervises the distribution using a **soft two-hot** cross-entropy.
Given a continuous target $\hat{d}$, it is interpolated between the
two nearest integer bins:

$$
k_L = \lfloor \hat{d} \rfloor, \qquad k_R = k_L + 1
$$

$$
w_L = k_R - \hat{d}, \qquad w_R = \hat{d} - k_L
$$

$$
\mathcal{L}_{\text{DFL}} = w_L \cdot \text{CE}(\mathbf{z},\; k_L) + w_R \cdot \text{CE}(\mathbf{z},\; k_R)
$$

```
  Soft two-hot encoding for target d̂ = 4.3

  Weight
   1.0 │
       │
   0.7 │         ██                        w_L = 5 - 4.3 = 0.7
       │         ██
       │         ██
   0.3 │         ██ ██                     w_R = 4.3 - 4 = 0.3
       │         ██ ██
   0.0 └──────── ██ ██ ──────────
       0    1    2    3    4    5    6    7
                 kL=4 kR=5

  CE loss is weighted: 0.7 × CE(z, bin4) + 0.3 × CE(z, bin5)
```

For this to be valid, we need $k_R \leq R$, which requires:

$$
\boxed{\hat{d} < R}
$$

In practice, a small epsilon $\varepsilon$ is subtracted to stay
strictly below $R$:

$$
\hat{d}_{\text{clamped}} = \text{clamp}(\hat{d},\; 0,\; R - \varepsilon)
$$

---

## The Bug

### What the code had

```python
# nanodet_head.py, _compute_loss()
dist_targets[pos_inds] = (raw_distances / pos_strides).clamp(
    min=0, max=self.reg_max - 1 - 0.01          # ← BUG
)
```

With `reg_max = 7` this evaluates to:

$$
\hat{d}_{\max} = 7 - 1 - 0.01 = \mathbf{5.99}
$$

### What the official NanoDet-Plus does

```python
target_corners = target_corners.clamp(min=0, max=self.reg_max - 0.1)
```

$$
\hat{d}_{\max} = 7 - 0.1 = \mathbf{6.9}
$$

### Correct fix

```python
dist_targets[pos_inds] = (raw_distances / pos_strides).clamp(
    min=0, max=self.reg_max - 0.01               # ← FIXED
)
```

$$
\hat{d}_{\max} = 7 - 0.01 = \mathbf{6.99}
$$

The stray `- 1` was an off-by-one error that removed an entire bin's
worth of regression range.

```
  Usable DFL bin range comparison (reg_max = 7)

  Bin index:     0     1     2     3     4     5     6     7
                 ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤

  BUGGY clamp:   ├═══════════════════════════════╡░░░░░░░░░░│
                 0                             5.99         7
                                                 ▲
                                          max target = 5.99
                                          (loses bins 6-7 entirely!)

  FIXED clamp:   ├══════════════════════════════════════════╡│
                 0                                       6.99
                                                           ▲
                                                    max target = 6.99
                                                    (full range used)

  ═══ = usable range       ░░░ = wasted range
```

---

## Mathematical Impact

### Maximum Representable Distance

For a center prior at stride $s$, the maximum distance the model can
learn is:

$$
d_{\max} = \hat{d}_{\max} \cdot s
$$

| Stride $s$ | Buggy ($\hat{d}_{\max} = 5.99$) | Fixed ($\hat{d}_{\max} = 6.99$) | Lost range |
|:---:|:---:|:---:|:---:|
| 8  | 47.9 px  | 55.9 px  | 14.3% |
| 16 | 95.8 px  | 111.8 px | 14.3% |
| 32 | 191.7 px | 223.7 px | 14.3% |
| 64 | 383.4 px | 447.4 px | 14.3% |

In general the fractional loss is:

$$
\frac{6.99 - 5.99}{6.99} = \frac{1.00}{6.99} \approx \mathbf{14.3\%}
$$

### Maximum Representable Box Size

The maximum box width or height centred on a prior is
$2 \cdot d_{\max}$. For a 320 × 320 input:

| Stride | Buggy max box dim | Fixed max box dim |
|:---:|:---:|:---:|
| 8  | 95.8 px  | 111.8 px |
| 16 | 191.7 px | 223.7 px |
| 32 | 383.4 px (full image) | 447.4 px (full image) |
| 64 | 766.7 px | 894.7 px |

```
  Max reachable box at stride 16 (320×320 input)

  BUGGY (max = 5.99 × 16 = 95.8 px per side):
  ┌─────────────────────────────────────────────────┐ 320 px
  │                                                 │
  │         ┌───────────────────────┐               │
  │         │      95.8    95.8     │               │  Max box:
  │         │     ◄─────·─────►    │               │  191.7 × 191.7 px
  │         │           │  95.8    │               │
  │         │           ▼          │               │
  │         └───────────────────────┘               │
  │                                                 │
  └─────────────────────────────────────────────────┘

  FIXED (max = 6.99 × 16 = 111.8 px per side):
  ┌─────────────────────────────────────────────────┐ 320 px
  │                                                 │
  │     ┌───────────────────────────────┐           │
  │     │       111.8    111.8          │           │  Max box:
  │     │      ◄──────·──────►         │           │  223.7 × 223.7 px
  │     │             │ 111.8          │           │
  │     │             ▼                │           │  +32 px wider
  │     └───────────────────────────────┘           │  +32 px taller
  │                                                 │
  └─────────────────────────────────────────────────┘
```

At stride 8 (the finest level, which handles most small-to-medium
objects), the maximum detectable box width/height drops from 111.8 to
95.8 — a loss of 16 pixels. At stride 16, the loss is 32 pixels. Any
ground-truth box whose edge extends beyond these limits will have its
DFL target **silently clamped**, teaching the model that the correct
distance is smaller than it really is.

---

## Why This Causes Systematic Under-Prediction

### Conflicting Loss Gradients

The total box regression loss has two components:

$$
\mathcal{L}_{\text{box}} = \mathcal{L}_{\text{GIoU}} + \lambda_{\text{DFL}} \cdot \mathcal{L}_{\text{DFL}}
$$

- **GIoU loss** operates on decoded pixel-space boxes and wants the
  prediction to match the **true** ground-truth box, pushing the model
  to predict larger distances.

- **DFL loss** operates on the normalised distribution targets and
  supervises toward the **clamped** target (max 5.99 instead of 6.99),
  pulling the model to predict smaller distances.

For any positive sample whose true normalised distance
$\hat{d}_{\text{true}} > 5.99$, the two losses produce **opposing
gradients**:

$$
\nabla_{\text{GIoU}}: \quad \text{"increase } \tilde{d} \text{ toward } \hat{d}_{\text{true}} \text{"}
$$

$$
\nabla_{\text{DFL}}: \quad \text{"decrease } \tilde{d} \text{ toward 5.99"}
$$

```
  Gradient Tug-of-War (for a sample with true d̂ = 6.5)

                          Buggy DFL target
                          (clamped to 5.99)     True target
                                │                   │
  0 ──────────────────────── 5.99 ─── ? ─── 6.5 ───── 7  (d̂ axis)
                                │     ▲       │
                     DFL pulls ◄┘     │       └► GIoU pulls
                     toward 5.99      │          toward 6.5
                                      │
                              Model converges
                              here: ~6.2 (compromise)
                              → box is too small!

  With the fix (DFL target = 6.5, matching GIoU):

  0 ──────────────────────────────── 6.5 ──────────── 7  (d̂ axis)
                                      ▲
                               Both losses agree!
                               → box is correct
```

The model converges to a **compromise** between the two, resulting in
a predicted distance that is less than the true distance:

$$
5.99 \;<\; \tilde{d}_{\text{converged}} \;<\; \hat{d}_{\text{true}}
$$

This means the predicted box is **systematically smaller** than ground
truth.

### Why the 0.5x Model Is Most Affected

```
  Model capacity vs bias resistance

  Backbone      Channels          Params     Bias resistance
  ─────────────────────────────────────────────────────────
  0.5x          [48, 96, 192]     ~0.49M     LOW  ████░░░░░░
  1.0x          [116, 232, 464]   ~1.17M     MED  ██████░░░░
  1.5x          [176, 352, 704]   ~2.44M     HIGH ████████░░

  Lower capacity → harder to resolve conflicting DFL/GIoU gradients
                 → compromise point sits further from the true target
                 → more visible bbox shrinkage
```

1. **Lower model capacity.** The 0.5x ShuffleNetV2 backbone has channels
   `[48, 96, 192]` compared to `[116, 232, 464]` for 1.0x. With fewer
   parameters, the model cannot easily learn to ignore the DFL signal
   and rely solely on GIoU. The compromise point sits further from the
   true target.

2. **Reduced auxiliary head.** The 0.5x variant uses only 2 stacked
   convs in the AGM auxiliary head (vs 4 for larger backbones). This
   weakens the assignment guidance, so the model receives noisier
   positive/negative supervision, amplifying any bias in the regression
   targets.

3. **Proportionally larger objects.** With a 320×320 input, the same
   real-world object occupies a larger fraction of the image. More
   positive samples hit the clamping ceiling, so the bias affects a
   larger fraction of the training signal.

---

## Worked Example

Consider a ground-truth box assigned to a center prior at stride 16:

- Center prior: $(p_x, p_y) = (160, 128)$
- GT box: $(x_1, y_1, x_2, y_2) = (48, 32, 272, 240)$

```
  320 px
  ┌────────────────────────────────────────────────────────┐
  │  (48,32)                                               │  0
  │   ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                    │
  │   ╎            t=96              ╎                    │
  │   ╎              ↑               ╎                    │  32
  │   ╎   l=112  ←── P(160,128) ──→ ╎ r=112              │  128
  │   ╎              ↓               ╎                    │
  │   ╎            b=112             ╎                    │
  │   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘                    │  240
  │                               (272,240)                │
  │                                                        │
  └────────────────────────────────────────────────────────┘  320
  0   48                          272                    320
```

**True distances (pixels):**

$$
l = 160 - 48 = 112, \quad t = 128 - 32 = 96
$$

$$
r = 272 - 160 = 112, \quad b = 240 - 128 = 112
$$

**True normalised distances** ($\div s = 16$):

$$
\hat{l} = 7.0, \quad \hat{t} = 6.0, \quad \hat{r} = 7.0, \quad \hat{b} = 7.0
$$

**Buggy clamped targets** (max = 5.99):

$$
\hat{l}' = 5.99, \quad \hat{t}' = 5.99, \quad \hat{r}' = 5.99, \quad \hat{b}' = 5.99
$$

**Fixed clamped targets** (max = 6.99):

$$
\hat{l}' = 6.99, \quad \hat{t}' = 6.0, \quad \hat{r}' = 6.99, \quad \hat{b}' = 6.99
$$

```
  Decoded box comparison (stride 16)

  BUGGY — all four targets clamped to 5.99:
   width  = (5.99 + 5.99) × 16 = 191.7 px   (GT = 224)  → 32.3 px short!
   height = (5.99 + 5.99) × 16 = 191.7 px   (GT = 208)  → 16.3 px short!

  FIXED — only l, r, b clamped to 6.99 (t = 6.0 unchanged):
   width  = (6.99 + 6.99) × 16 = 223.7 px   (GT = 224)  →  0.3 px off
   height = (6.0  + 6.99) × 16 = 207.8 px   (GT = 208)  →  0.2 px off
```

Decoded box width comparison:

- **True:** $(l + r) \times 16 = (7.0 + 7.0) \times 16 = 224$ px
- **Buggy target:** $(5.99 + 5.99) \times 16 = 191.7$ px — **32.3 px too small** (14.4%)
- **Fixed target:** $(6.99 + 6.99) \times 16 = 223.7$ px — **0.3 px off** (0.1%)

---

## End-to-End Data Flow (with bug location)

```
  ┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
  │  Input Image │────►│  Backbone   │────►│  GhostPAN (FPN)  │
  │  320 × 320   │     │ ShuffleNetV2│     │                  │
  └─────────────┘     │    0.5x     │     └────────┬─────────┘
                      └─────────────┘              │
                                          Multi-scale features
                                        (stride 8, 16, 32, 64)
                                                   │
                                                   ▼
                                       ┌───────────────────────┐
                                       │  NanoDetPlusHead       │
                                       │  Per-scale conv layers │
                                       │  gfl_cls output layer  │
                                       └───────────┬───────────┘
                                                   │
                                    ┌──────────────┴──────────────┐
                                    │                             │
                              cls logits                   reg logits
                           [B, N, num_cls]           [B, N, 4×(reg_max+1)]
                                    │                             │
                                    │                    ┌────────┴────────┐
                                    │                    │    Integral     │
                                    │                    │ softmax → E[x]  │
                                    │                    └────────┬────────┘
                                    │                             │
                                    │                    distance × stride
                                    │                             │
                                    │                    ┌────────┴────────┐
                                    │                    │  distance2bbox  │
                                    │                    │  point ± dist   │
                                    │                    └────────┬────────┘
                                    │                             │
                                    ▼                             ▼
                             ┌────────────┐              ┌──────────────┐
                             │  QFL Loss   │              │  GIoU Loss   │
                             │ (cls score) │              │ (decoded box)│
                             └────────────┘              └──────────────┘
                                                                 │
                                    ┌────────────────────────────┤
                                    │                            │
                                    ▼                            │
                        ┌─────────────────────┐                  │
                        │  DFL Loss            │                  │
                        │  (distribution bins) │                  │
                        │                      │                  │
                        │  ★ BUG WAS HERE ★   │                  │
                        │  Target clamped to   │                  │
                        │  5.99 instead of     │                  │
                        │  6.99 — DFL trains   │◄─── Conflicting
                        │  model toward too-   │     gradients on
                        │  small distances     │     the same params
                        └─────────────────────┘
```

---

## Fix Applied

**File:** `src/models/head/nanodet_head.py`, method `_compute_loss()`

```diff
- dist_targets[pos_inds] = (raw_distances / pos_strides).clamp(
-     min=0, max=self.reg_max - 1 - 0.01
- )
+ dist_targets[pos_inds] = (raw_distances / pos_strides).clamp(
+     min=0, max=self.reg_max - 0.01
+ )
```

**Action required:** Retrain all affected models (especially 0.5x)
after the fix so they benefit from the corrected regression targets.

---

## References

- Li, X. et al. *Generalized Focal Loss: Learning Qualified and
  Distributed Bounding Boxes for Dense Object Detection.* NeurIPS 2020.
- Li, X. et al. *Generalized Focal Loss V2: Learning Reliable
  Localization Quality Estimation for Dense Object Detection.* CVPR 2021.
- Official NanoDet-Plus: `nanodet/model/head/gfl_head.py`, target
  clamping at `self.reg_max - 0.1`.
