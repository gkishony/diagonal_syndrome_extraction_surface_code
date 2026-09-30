# Flag-aware decoding of the spatial Hadamard circuit — implementation and results

Implementation of the decoder proposed in `flag_aware_hadamard_decoder_proposal.md`,
plus the measurements used to evaluate it.

**Headline result.** Flag-awareness restores the full circuit distance at `k=1`
(effective distance `1 → 3`, identical to Tesseract) and lifts matching from the
paper's `2⌊k/2⌋+1` to `2⌈k/2⌉+1` in general — one step better, still one step short
of Tesseract's `2k+1`. The gap it closes and the gap it leaves have the same cause:
single faults whose decomposed pieces land in the same matching problem and are
therefore charged twice (§4). Two later findings sharpen this. Only one of the
decoder's two mechanisms does any work — pricing an unheralded hook as two faults,
which is the proposal's headline idea, turns out to be inert at every `k` (§4.5).
And with `flag_config=all` the `k=1` deficit disappears for *plain* matching, so the
decoder's `k=1` win is really a repair of the sparser `partial` flag set (§7).

---

## 1. What was built

### `flag_aware_decoder.py`

`FlagAwareMatching` starts from **pymatching's own matching graph** — which preserves
its logical masks and, with `enable_correlations`, the decomposition metadata its
two-pass decoder needs — and then rewrites only the weights of the *flag-free* pieces
of flagged hyperedges.

The spatial Hadamard construction measures flag qubits that herald the hook errors of
the stretched stabilizers. Such a mechanism has detector signature `H_i = G_i ⊕ F_i`,
with `F_i` the flag detectors. Stim decomposes it into graphlike pieces, some carrying
flag detectors and some not, and a matching decoder treats those pieces as independent
errors. The flag-free part of a hook therefore stays available as a weight-`p` shortcut
*even in shots where the flags stayed quiet*, and that is what costs the decoder its
effective distance.

The fix is to price those pieces conditionally:

- by default, a flag-free piece costs the mechanism **plus the extra fault that would
  have to silence the flags**, which restores the two-fault cost of an unheralded hook;
- in shots where the flags did all fire, the piece is restored to the bare single-fault
  weight.

Three strategies are built on that graph:

| mode | behaviour |
|---|---|
| `reweight` | one gated matching call |
| `iterative` | scored commitment of heralded mechanisms, then a clean pass |
| `hyperedge` | the same scored commitment, but *any* split mechanism may be proposed, flagged or not |

### Supporting tools

- **`test_flag_aware_decoder.py`** — 8 unit tests (all passing). A toy 4-detector DEM
  with one flagged hook, plus a 24-detector ring DEM where flagged hooks halve the
  matching distance. Covers model parsing, reduction to plain/correlated pymatching when
  there are no flags, a hook costing two faults unflagged and one flagged, correct and
  incorrect correction with and without the flag, agreement between modes, batch versus
  single-shot equivalence, and flag-awareness beating plain matching on the ring.
- **`benchmark_flag_aware_decoder.py`** — sinter sweep over `k` and `p` with an
  effective-distance fit, a CSV, and a plot. Checkpointed, so an interrupted run resumes.
- **`measure_decoder_fault_distance.py`** — implements §19 of the proposal. Enumerates
  fault sets of increasing weight from the DEM and reports the first weight that fools
  each decoder. A decoder first failing at weight `t+1` has `d_eff = 2t+1`.
- **`ablate_flag_aware_decoder.py`** — turns the silencing penalty and the shared
  heralded weight off one at a time, across `k` and flag configurations, and counts
  mis-decoded fault sets. This is what shows only one of the two matters (§4.5).

### Repo fixes needed along the way

- `spatial_hadamard_manual_construction.py`, `patch_rotation_manual.py`,
  `memory_from_plaquettes.py`: added `reschedule_measurements=False` at the four
  `layer_tree.generate_circuit(...)` call sites. The installed tqec reschedules
  measurements by default, which breaks the repo's custom and empty plaquettes
  (`TQECError: No Moment instance scheduled at the provided schedule 0`).
- `compact_circuit.py`: re-emit `DETECTOR` instructions with `tag=orig_inst.tag`.
  Without this the flag tags were dropped during compaction and the decoder saw
  **zero** flag detectors.

### How to run

```bash
./venv/bin/python -m pytest test_flag_aware_decoder.py

# fault-count distance; exhaustive up to --max-weight, or sampled
./venv/bin/python measure_decoder_fault_distance.py --k 1 --max-weight 2 --include-tesseract
./venv/bin/python measure_decoder_fault_distance.py --k 3 --max-weight 2 --sample 120000

# logical error rate sweep with an effective-distance fit
./venv/bin/python benchmark_flag_aware_decoder.py --k-values 1 2 3 \
    --decoders pymatching correlated_pymatching flag_aware_reweight flag_aware tesseract
./venv/bin/python benchmark_flag_aware_decoder.py --plot-only

# which half of the flag conditioning actually matters (§4.5)
./venv/bin/python ablate_flag_aware_decoder.py --k-values 1 2 --flag-configs partial all
```

---

## 2. Results

All measurements use axis `y`, interface-only noise, and `flag_config=partial` unless
stated otherwise; §7 repeats the key ones with `flag_config=all`.

### 2.1 Fault counting (exhaustive)

The number of physical faults needed to fool each decoder. This is the proposal's
"most important conceptual test" and does not depend on a Monte Carlo slope fit.

| k | circuit size | plain matching | correlated matching | flag-aware | tesseract | true distance |
|---|---|---|---|---|---|---|
| 1 | 60 detectors, 12 flags, 373 mechanisms | 1 | 1 | **3** | 3 | 3 |
| 2 | 270 detectors, 30 flags, 1127 mechanisms | 3 | 3 | 3 | **≥5** | 5 |
| 3 | 728 detectors, 56 flags, 2291 mechanisms | 3 | 3 | **≥5** | — | 7 |

Details:

- **k=1** was exhaustive to weight 2. Plain and correlated matching are both fooled by a
  *single* physical fault. Every flag-aware variant survives all of them, and Tesseract
  fails on the very same weight-2 example — so the flag-aware decoder is saturating the
  circuit's true distance here, not merely improving on matching.
- **k=2** was exhaustive to weight 2 (635,628 fault sets). Tesseract survived all of
  them, which pins the ceiling at `d_eff ≥ 5` and confirms the matching family's cap of
  3 is a real decoder limitation rather than a quirk of the circuit.
- **k=3** sampled 120,000 sets per weight. Both flag-aware variants survived everything;
  both matching baselines were fooled at weight 2.

### 2.2 Monte Carlo sweep

Effective distance fitted from the low-`p` slope of `p_logical ∝ p^(t+1)`:

| decoder | k=1 | k=2 | k=3 |
|---|---|---|---|
| pymatching | 1.90 | 3.00 | 3.10 |
| correlated pymatching | 1.49 | 3.02 | 5.21 |
| flag-aware (reweight) | 3.20 | 3.06 | 5.44 |
| flag-aware (iterative) | 2.71 | 3.00 | 5.07 |
| tesseract | 2.83 | 5.12 | 7.09 |
| *ideal* `2k+1` | *3* | *5* | *7* |

Logical error rate at `p = 0.001` (observed errors in parentheses):

| decoder | k=1 | k=2 | k=3 |
|---|---|---|---|
| pymatching | 3.64e-3 (150) | 1.11e-3 (150) | 7.99e-4 (150) |
| correlated pymatching | 3.03e-3 (153) | 6.44e-4 (150) | 4.33e-5 (13) |
| flag-aware (reweight) | 8.41e-4 (150) | 5.46e-4 (150) | 1.00e-5 (3) |
| flag-aware (iterative) | 8.74e-4 (150) | 5.37e-4 (150) | 3.67e-5 (11) |
| tesseract | 6.12e-4 (151) | 1.67e-5 (5) | 0 (0) |

Flag-awareness beats plain matching by 4.3× at k=1, 2.1× at k=2, and roughly 20× at
k=3, and beats correlated matching by 3.6× at k=1.

Data in `benchmark_data/flag_aware_decoder.csv`, plot in
`figures/flag_aware_decoder.png`.

**Caveat.** The 300k-shot cap left the k=3 low-`p` points with only 3–13 logical errors
(zero for Tesseract), so the k=3 slopes carry real uncertainty. The k=3 claim rests on
the exhaustive fault counting, not on the fit. The k=1 and k=2 rows have ~150 errors per
point and are solid.

### 2.3 Agreement between the two methods

The two independent measurements agree everywhere, which is the main reason to trust
either. Plain matching reproduces the paper's `2⌊k/2⌋+1` = 1, 3, 3 in both. Flag-aware
gives 3, 3, 5 in both — i.e. `2⌈k/2⌉+1`.

---

## 3. Which proposal outcome held

The proposal's **Outcome A**: direct flag-conditioned reweighting is sufficient. It
recovers the distance on its own, and iterative refinement adds nothing on top.

In fact the first greedy version of the iterative mode actively *cost* distance
(`d_eff = 1` at k=1), because it committed any heralded mechanism whose flag-free pieces
appeared in the current solution. It only stopped doing harm once commitment was gated
on a hypothesis strictly lowering the total solution weight.

---

## 4. Why it stops short of `2k+1`

A matching graph cannot express "these two edges are one fault". Once stim decomposes a
hyperedge, each piece is priced independently, and that goes wrong in two opposite ways:

- **undercharging** — a piece is usable on its own for the price of one fault, even when
  no single fault produces it, which invents a shortcut that does not physically exist;
- **overcharging** — a genuine single fault is billed once per piece, so the true
  explanation looks more expensive than it is and can lose to a wrong one.

Decomposition is not harmful *per se*. The reference case is Y errors in an ordinary
surface code, which decompose into an X and a Z piece and cost nothing in distance. Two
structural properties make them safe, and measuring both is what identifies the problem
here. Numbers below compare stim's `rotated_memory_z` at d=5 against the Hadamard DEM at
k=2.

### 4.1 Undercharging: only the flagged hooks, and they are fixed

A piece is harmless if some standalone single fault produces exactly that detector set,
because then the edge matching uses is real rather than invented.

| | pieces with a standalone counterpart | phantom pieces (no such fault) |
|---|---|---|
| surface code memory d=5 | 3204 | **0** |
| spatial Hadamard k=1 | 755 | **0** |
| spatial Hadamard k=2, `partial` flags | 2337 | 78 |
| spatial Hadamard k=2, `all` flags | 2178 | 68 |

The surface code has no phantom pieces at all, which is precisely why decomposed Y
errors are safe. The Hadamard circuit has some from k=2 on, and **every one of them sits
on a flagged mechanism** — zero on unflagged ones. These are the stretched-stabilizer
hooks, and they are exactly what the silencing penalty is meant to remove.

They turn out not to matter, though. At `k=1` there are none at all, yet plain matching
still fails at a single fault there; and the ablation in §4.5 shows that switching the
silencing penalty off costs nothing at any `k`. Undercharging is real but never
decisive — the whole effect, in both directions, is overcharging.

### 4.2 Overcharging: the part that remains

A double charge is harmless if the pieces land in *separate* matching problems, since
then each problem sees only one edge for the fault.

| | pieces in **different** components | pieces in the **same** component |
|---|---|---|
| surface code memory d=5 | 1341 | 110 |
| spatial Hadamard k=2 | 810 | 81 (66 flagged, 15 unflagged) |

The surface code splits into exactly two detector components — the X and Z sublattices —
and a Y error puts one piece in each, so no matching problem ever double-charges it. The
Hadamard interface deliberately swaps X and Z, fusing the two sublattices near the
interface, so there a single fault can put both pieces into the same problem and the
double charge becomes real.

That is the canonical residual failure at k=2. The true event is two faults, `D[9]` and
the split mechanism `D[9,48,279]`, neither touching a flag. Representing it needs three
graph edges — `(9,B)`, `(48,B)`, `(9,279)` — so two faults are billed as three, and the
true explanation loses to a genuine three-edge path that is cheaper and carries the
logical observable:

```
pymatching      -> L[0]   total weight 16.58
        48 --    B   w=5.93   fault_ids=[]
       280 --  279   w=5.03   fault_ids=[]
       280 --    B   w=5.63   fault_ids=[0]
flag_aware reweight   -> L[0]     (wrong)
flag_aware iterative  -> L[0]     (wrong)
flag_aware hyperedge  -> L[]      (correct)
```

No flag is involved and no phantom edge is involved; the true explanation is simply
overpriced. Flag conditioning cannot help with this particular shot, and adding more
flags does not fix the class either (§7). Hyperedge commitment is the right lever and does
fix this shot, but it only proposes mechanisms owning an edge the current solution
already picked, and in most surviving failures the correct hyperedge appears nowhere in
the wrong solution.

### 4.3 Ruled out: ambiguous edges

Two mechanisms with identical detectors but different observables would collapse onto one
edge, making the logical mask a coin flip. This does not occur at all: of 1024 distinct
mechanism signatures and 245 graph edges, **zero** carry conflicting observables.

### 4.4 Remaining headroom

On 20,000 random weight-2 fault sets at k=2 — a full-distance decoder would get all of
them right. Every failure, for every decoder, used a piece of a split mechanism.

| decoder | weight-2 sets decoded wrong |
|---|---|
| pymatching | 1.62% |
| correlated pymatching | 0.93% |
| flag-aware (reweight) | 0.73% |
| flag-aware (hyperedge commit) | 0.70% |

Flag-awareness removes more than half of plain matching's failures; the last 0.7% is what
stands between the decoder and `2k+1`.

**Caveat.** The surface code also has 110 intra-component split mechanisms and keeps its
distance, so landing in one component is necessary but not sufficient for harm. Whether
it actually costs distance depends on a cheaper wrong path existing nearby, which is a
quantitative property of the interface geometry rather than a clean structural rule. What
is solid is the empirical part: all phantom pieces are flagged, and every residual
weight-2 failure uses a piece of a split mechanism.

### 4.5 Which half of the conditioning does the work

`FlagAwareMatching` does two independent things to a flagged hyperedge, and
`ablate_flag_aware_decoder.py` switches them off one at a time. The **silencing
penalty** (`flag_silencing_probability`) prices an unheralded hook as the mechanism plus
the fault needed to keep the flags quiet — this is the proposal's central idea, and the
fix for undercharging. **Shared weight** (`share_heralded_weight`) spreads a heralded
mechanism as `p^(1/n)` over its `n` pieces so that using all of them costs one fault
rather than `n` — the fix for overcharging.

Columns are mis-decoded fault sets: all single faults exhaustively, plus a random sample
at the weight where failures first appear (30k sets at k=1 and k=2, 1.5k at k=3, ties
excluded).

| configuration | k=1 `partial` | k=1 `all` | k=2 `partial` | k=2 `all` | k=3 `partial` |
|---|---|---|---|---|---|
| | w1 / w2 | w1 / w2 | w1 / w2 | w1 / w2 | w1 / w3 |
| full decoder | **0** / 102 | 0 / 54 | 0 / 209 | 0 / 86 | 0 / 2 |
| shared weight OFF | **2** / 132 | 0 / 95 | 0 / 240 | 0 / 91 | 0 / 1 |
| silencing penalty OFF | **0** / 104 | 0 / 54 | 0 / 220 | 0 / 86 | 0 / 2 |
| both OFF | **1** / 132 | 0 / 95 | 0 / 250 | 0 / 91 | 0 / 1 |

Read the bolded `k=1 partial` column first: it is the only place where the distance
itself moves. Turning off shared weight costs two single-fault failures and drops
`d_eff` from 3 to 1. Turning off the silencing penalty costs nothing. **The entire
distance gain is the overcharge fix; the proposal's headline mechanism contributes
nothing measurable anywhere.** It is not a no-op — it does shift the affected weights by
0.5–1.2 units — it just never changes an outcome, because every interface piece it
targets already has a standalone counterpart at `p ≈ 3–4e-3`, far above the `6.7e-4`
hook it is competing with.

Away from the distance threshold the silencing penalty is worth at most 1–5% of the
residual failures (209 vs 220 at k=2), which is within the run-to-run spread. It could
be dropped, which would make the decoder simpler and remove a per-shot gating pass.

### 4.6 Mechanisms carrying more than one flag

Heralding is all-or-nothing: a mechanism is recognised only if *every* one of its flags
fired, so a two-flag hook with just one flag lit stays unheralded and pays the silencing
penalty. That rule is also what makes it safe to file mechanisms under `min(flags)`
alone. In the commitment modes an accepted mechanism consumes all of its flags, so a
second mechanism cannot reuse the same evidence.

Nothing in this circuit carries more than two flags, and two-flag mechanisms are a small
minority — 18 of 201 flagged mechanisms at k=1 and 50 of 621 at k=2, the same counts for
either flag density. The one place they behave differently is the silencing price:
`_silencing_probability` looks for a single fault flipping exactly that pair, and only
if none exists charges the product of the two single-flag probabilities, which roughly
doubles the penalty weight. That fallback is used for 12 of 201 mechanisms at k=1 and 40
of 621 at k=2 with `partial` flags; with `all` flags the DEM contains a direct two-flag
silencing fault for every pattern that occurs, so it never triggers. No pattern is ever
left forbidden.

**They do not matter.** Disabling heralding for multi-flag mechanisms entirely leaves
single-fault failures at zero and barely moves the rest (per 20k sampled weight-2 sets:
743 → 747 at k=1 `partial`, 147 → 148 at k=2 `partial`, unchanged at 422 and 47 with
`all` flags). The two single faults that shared weight rescues at k=1 are both on
single-flag mechanisms, so the distance result does not depend on any of this.

**The gate is loose in both directions, though.** Spurious heralding is common: in
weight-2 shots a two-flag mechanism is declared heralded more often when it did *not*
occur than when it did — 5218 versus 2958 at k=1 `partial` — because two independent
faults each lighting one flag of the pair look exactly like the pair. It is harmless
here because heralding only makes those pieces cheaper and matching must still explain
the rest of the syndrome. The mirror-image error also exists and is invisible in these
counts: two faults sharing a flag detector cancel it, and a mechanism that really did
occur then goes unheralded and gets penalised.

---

## 5. Routes to full distance

**Belief-matching — tried, and it does not work here.** The idea was that belief
propagation on the hyperedge graph would couple the pieces of a mechanism and fix both
failure directions at once. Measured with the `beliefmatching` package, it does not:

| | k=1 (true distance 3) | k=2 (true distance 5) |
|---|---|---|
| belief matching | first fails at weight **1**, `d_eff = 1` | weight 2, `d_eff = 3` |
| flag-aware | weight 2, `d_eff = 3` | weight 2, `d_eff = 3` |

A single physical fault fools it at k=1, no better than plain matching. This is not a
tuning artifact: both `product_sum` and `minimum_sum` at 5, 20, 30, 100 and 200 BP
iterations all fail at weight 1. BP shifts probabilities in the right direction — its raw
error rate is competitive (§5.1) — but it never makes a phantom edge *forbidden*, so one
well-placed fault still wins. Hard flag conditioning does forbid it, which is exactly why
the flag-aware decoder reaches `d_eff = 3` at k=1 and belief matching does not.

It is also not cheap. Throughput at p=1e-3, single-threaded, 20,000 shots:

| decoder | k=1 | k=2 | k=3 |
|---|---|---|---|
| pymatching | 10,900,000/s | 2,740,000/s | 1,210,000/s |
| correlated pymatching | 5,090,000/s | 1,310,000/s | 577,000/s |
| flag-aware (reweight) | 62,000/s | 6,600/s | 1,800/s |
| belief matching | 32,000/s | 4,400/s | 1,160/s |
| tesseract | 2,024/s | 173/s | 25/s |

Logical errors out of 20,000 shots at p=1e-3 — k=1: 83 pymatching, 66 correlated,
17 flag-aware, 28 belief matching; k=2: 22 / 10 / 7 / 5; k=3: 18 / 4 / 0 / 1. Comparable
to flag-aware on raw error rate, clearly worse at k=1, and worse on distance everywhere.

Two caveats on the timings. The flag-aware decoder sends any shot with no flags lit
straight to pymatching's batch decoder, and at p=1e-3 that is most shots, so its
advantage shrinks at larger `p`. And `beliefmatching` loops in Python per shot even though
BP itself is compiled, so a tighter implementation would be faster.

**Fix and widen the hypothesis search — done, and it helps but is not enough.** Two
defects in `_commit_hyperedges` followed directly from §4.2 and have both been fixed:

- *Inconsistent objective.* A candidate was scored as the mechanism's weight plus the
  re-matched residual, but compared against a baseline that summed raw edge weights —
  so the baseline was billed per piece while the candidate was billed per fault. Both
  sides now use `_corrected_cost`, which charges each split mechanism once wherever all
  its pieces are present, discounting each edge at most once.
- *Candidate generation too narrow.* Only mechanisms owning an already-selected edge were
  proposed. That is backwards for overcharging: the wrong solution that beats a true
  mechanism usually contains none of its pieces. Candidates are now also drawn from the
  mechanisms touching any lit detector.

Measured on 5,000 random weight-2 fault sets at k=2:

| decoder | wrong |
|---|---|
| pymatching | 1.76% |
| correlated pymatching | 0.90% |
| flag-aware (reweight) | 0.78% |
| flag-aware (hyperedge, before the fix) | 0.70% |
| **flag-aware (hyperedge, after the fix)** | **0.48%** |

Raising `max_candidates` from 8 to 64 only reaches 0.46%, so generation is no longer the
bottleneck — greedy one-at-a-time acceptance is. Allowing locally-worse commits via a
negative `commit_margin` was tested as cheap lookahead: a small slack helps slightly
(0.40% at −3) but more slack is much worse (1.34% at −14), because it starts committing
bad hypotheses. Real lookahead means keeping several branches alive and comparing final
costs, which is Stage 5 beam search and no longer a small change. The caveat is that as
the beam widens this converges toward reimplementing Tesseract.

`d_eff` at k=2 is still 3: 0.48% is a large improvement but not zero.

**Escalate to an exact decoder on suspicious shots.** The remaining credible route to a
*guaranteed* `2k+1`. Matching can tell when it is in trouble: its solution uses an
orphaned piece of a split mechanism, or the complementary gap between the two logical
classes is small. Run flag-aware matching on everything and hand only the suspicious
shots to Tesseract. The throughput table above makes this look affordable — Tesseract at
173 shots/s on ~1% of k=2 shots is a modest surcharge on a 6,600 shots/s decoder. Open
questions: the escalation rate at realistic `p`, and whether the trigger has good enough
recall.

**Ensembling flag-aware with belief matching** is worth a look, since the two fail on
different fault sets. The gating measurement is the oracle rate — the fraction of
weight-2 sets that *every* candidate decoder gets wrong. If that is zero, a selector
based on `_corrected_cost` could reach full distance; if not, ensembling cannot. Not yet
measured.

---

## 6. Methodology notes

Two measurement details that changed the numbers materially and are worth keeping:

- **Tie filtering in the distance test.** The first version counted any disagreement with
  the injected faults as a failure. That over-reports, because some syndromes have two
  equally-cheap explanations in different logical classes, where either answer is
  legitimate. `ExplanationIndex` now checks for such a competing explanation at the same
  weight and only counts genuine distance losses. This is what makes the exhaustive
  counts and the Monte Carlo slopes agree.
- **Robust slope fit.** Fitting `d_eff` from the two lowest `p` points is very noisy when
  the lowest point has few observed errors. The fit now drops points with fewer than 10
  errors and takes a least-squares line through the three lowest surviving `p` values.

---

## 7. Flag density: `partial` versus `all`

Everything above used `flag_config=partial`, which flags only some of the stretched
stabilizers. `all` flags every one of them, which roughly doubles the flag detectors
(k=1: 12 → 18; k=2: 30 → 50; k=3: 56 → 98) at the cost of extra measurement qubits.

**At k=1 the deficit disappears entirely, for every decoder.** Exhaustively enumerating
all fault sets up to weight 2, `pymatching`, `correlated_pymatching`, all three
flag-aware modes and Tesseract *all* first fail at weight 2, i.e. `d_eff = 3 = 2k+1`.
Plain matching gets the full distance for free, and the flag-aware decoder has nothing
left to repair. This is confirmed end to end by the sinter sweep:

| decoder | k=1 `partial` | k=1 `all` | k=2 `partial` | k=2 `all` |
|---|---|---|---|---|
| pymatching | 1.90 | **2.89** | 3.00 | 2.97 |
| correlated pymatching | 1.49 | **2.95** | 3.02 | 3.25 |
| flag-aware (reweight) | 3.20 | 2.88 | 3.06 | 3.44 |

(fitted `d_eff`; true distance is 3 and 5. `benchmark_data/flag_aware_decoder_allflags.csv`.)

**At k=2 it does not restore the distance.** The exhaustive weight-2 scan with `all`
flags still finds a failure for `pymatching`, `correlated_pymatching` and
`flag_aware_reweight` alike — `d_eff = 3` against a true distance of 5. Structurally
this is expected: full flags only take the phantom pieces from 78 down to 68, and §4
already established that phantom pieces are not what costs the distance.

What full flags *do* buy at k=2 is a substantially lower error rate at fixed distance:
the weight-2 failure count drops from 209 to 86 per 30k sets (§4.5) and the logical
error rate at the lowest sampled `p` falls from `1.14e-3` to `3.80e-4` between plain
matching and flag-aware. The ordering of decoders is preserved, just compressed.

**Net reading.** The `partial` configuration leaves a hole that either more flags or a
flag-aware decoder can plug, and at k=1 the two are interchangeable — the decoder is the
cheaper of the two, since it needs no extra qubits. Neither reaches `2k+1` from k=2 on,
because the obstruction there is the double-charging of split mechanisms, which more
flags cannot address.
