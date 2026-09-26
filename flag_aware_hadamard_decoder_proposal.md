# Flag-Aware Syndrome-Conditioned Matching for the Spatial Hadamard Circuit

## 1. Motivation

In the spatial Hadamard construction of *Surface code off-the-hook: diagonal syndrome-extraction scheduling* ([arXiv:2602.09099](https://arxiv.org/abs/2602.09099)), stretched stabilizers at the Hadamard interface can produce dangerous hook errors.

Without additional information, these hooks act as shortcuts along logical operators and reduce the effective distance of the interface.

Flag measurements are introduced precisely to identify these dangerous correlated faults. At the circuit level, the flags restore the full fault-tolerant distance. However, standard MWPM, and even correlated matching as currently implemented in PyMatching, do not fully exploit this information and can still exhibit the reduced effective-distance scaling.

The decoding problem is therefore:

> How can we use measured flag information to recover the full circuit distance while retaining a decoder that is close to MWPM in speed and scalability?

The central proposal is a **flag-aware, syndrome-conditioned matching decoder** that treats flags as side information about a small set of possible physical hyperedges, rather than as ordinary matching defects.

---

## 2. Structure of the problematic faults

Consider a dangerous physical hook fault $H_i$. Its detector signature can be written schematically as

$H_i = G_i \oplus F_i,$

where:

- $G_i$ is the ordinary detector pattern associated with the hook;
- $F_i$ is one or more flag detectors;
- the complete pattern $H_i$ arises from a **single physical fault** with probability $O(p)$.

The difficulty is that $H_i$ is generally not graphlike. It may touch more than two detectors, so it cannot directly be represented as a single edge in an ordinary matching graph.

A graph decomposition can instead represent it as

$G_i + F_i,$

but this representation loses the statement that the two pieces are consequences of the **same physical event**.

This distinction is essential. The decoder should understand that

$P(G_i \mid F_i=1)$

can be large, while

$P(G_i \mid F_i=0)$

may be much smaller.

---

## 3. Why ordinary MWPM fails

Ordinary MWPM works on a graph in which each elementary mechanism produces at most two detection events.

If the flagged hook hyperedge is decomposed into graphlike pieces and the correlations are ignored, MWPM may treat

- the dangerous hook syndrome $G_i$, and
- the flag syndrome $F_i$

as independent error mechanisms.

The dangerous spatial shortcut then remains available as an $O(p)$ matching edge. As a result, the matching graph can have a lower effective distance than the underlying flagged circuit.

The flags have restored the **physical circuit distance**, but the decoder has failed to preserve it.

---

## 4. Relation to correlated matching

PyMatching's correlated matching partially restores information about decomposed hyperedges.

Very schematically, it performs:

1. an initial MWPM pass;
2. identification of correlations associated with edges selected in that pass;
3. conditional reweighting of correlated edges;
4. a second MWPM pass.

For a simple decomposition

$H = A \oplus F,$

one can think of the logic as

$F$ selected in the first pass $\rightarrow$ make $A$ cheaper in the second pass.

This is already closely related to the proposed decoder.

However, there are two important limitations.

### 4.1 Conditioning is mediated by the first MWPM hypothesis

The physical observation is

$F=1.$

What we would ideally like to use is

$P(H_i \mid F=1, s_{\rm rest}),$

where $s_{\rm rest}$ is the surrounding syndrome.

Correlated matching instead conditions on components inferred by the **first matching solution**.

These are not necessarily equivalent. The first MWPM pass may explain the flag in a way that prevents the relevant correlation from being activated.

### 4.2 One flag may have several competing explanations

For the partial-flag Hadamard circuit, one measured flag can correspond to multiple physical faults.

Schematically,

$F_t = 1$

may be consistent with

- a left hook in round $t$;
- a right hook in round $t-1$ whose flag propagates forward;
- a flag measurement error;
- other nearby circuit faults.

Write two possible hook mechanisms as

$H_1 = G_1 \oplus F,$

$H_2 = G_2 \oplus F.$

Conditional reweighting can potentially make both $G_1$ and $G_2$ cheap.

But physically the measured flag should not be treated as independent evidence for both faults simultaneously.

This motivates making the interpretation of the flag an explicit decoding decision.

---

# 5. Proposed decoder

The proposed decoder combines three ideas:

1. **Direct conditioning on measured flag outcomes**;
2. **MWPM as a proposal mechanism for resolving ambiguous flag interpretations**;
3. **Iterative syndrome refinement after committing to a physical hyperedge**.

This can be viewed as a flag-specialized version of ghost-edge / iterative hyperedge decoding.

---

## 6. Preprocessing

Start from the **undecomposed detector error model**.

For every relevant physical fault mechanism $H_i$, record:

- its physical probability $p_i$;
- its full detector set $D_i$;
- its flag subset $F_i$;
- its non-flag detector component $G_i$;
- its logical observable mask $L_i$.

Thus,

$D_i = G_i \oplus F_i.$

Only a sparse subset of the detector error model needs special handling: the dangerous flagged faults near the Hadamard interface.

All ordinary graphlike faults remain in the standard matching graph.

---

## 7. Flag-gated ghost hypotheses

For a given shot, let $s$ be the observed detector syndrome.

For every active flag, collect the set of physical mechanisms compatible with it.

For example,

$F_t = 1 \Rightarrow \{H_1, H_2, H_3, \ldots\}.$

Each candidate $H_i$ has a graphlike component $G_i$.

Temporarily expose the corresponding $G_i$ components as **ghost hypotheses** to MWPM.

Crucially:

> A ghost edge is not interpreted as an independent physical error.

It is only a proposal that the complete physical mechanism $H_i$ occurred.

---

## 8. Syndrome refinement

Run MWPM on the working syndrome with the currently allowed ghost hypotheses.

Suppose MWPM selects a ghost component $G_i$.

Interpret this as evidence for the full physical fault

$H_i = G_i \oplus F_i.$

Commit to that hypothesis and update the working syndrome:

$s \leftarrow s \oplus H_i.$

Also update the accumulated logical correction:

$\lambda \leftarrow \lambda \oplus L_i.$

This has two effects:

1. the ordinary syndrome caused by the inferred hook is removed;
2. the corresponding flag is also removed.

The flag evidence has therefore been **consumed**.

This prevents the same flag from independently supporting several incompatible hook hypotheses.

---

## 9. Iteration

After committing one or more flagged mechanisms, rerun matching on the refined syndrome.

Repeat until one of the following occurs:

- no ghost hypothesis is selected;
- there are no active flags remaining;
- a small fixed iteration limit is reached.

Finally run ordinary MWPM with all artificial ghost shortcuts removed.

The final logical prediction is

$\lambda_{\rm final} = \lambda_{\rm committed} \oplus \lambda_{\rm MWPM}.$

The final matching graph should not contain any artificial low-cost edge that represents only part of a flagged hyperedge.

---

# 10. Simple partial-flag example

Suppose the flag detector $F_{t+1}$ fires.

There are two dangerous explanations:

$H_L(t+1) = G_L(t+1) \oplus F_{t+1},$

$H_R(t) = G_R(t) \oplus F_{t+1}.$

Here:

- $H_L(t+1)$ is a left hook in the current round;
- $H_R(t)$ is a right hook from the previous round whose flag information propagates to $F_{t+1}$.

Temporarily expose both

$G_L(t+1)$

and

$G_R(t)$

to MWPM.

Suppose the surrounding spacetime syndrome strongly favors $G_R(t)$.

The decoder commits

$H_R(t)$

and updates

$s \leftarrow s \oplus G_R(t) \oplus F_{t+1}.$

The flag $F_{t+1}$ is now zero.

Therefore the alternative hypothesis $H_L(t+1)$ can no longer reuse the same flag evidence.

This is the central advantage over treating all flag-conditioned hook edges as independently cheap.

---

# 11. An even simpler variant: syndrome-conditioned reweighting

Before implementing full iterative refinement, it is worth testing a simpler decoder.

Instead of waiting for a first MWPM pass to activate correlations, directly use the **observed flag syndrome** to modify the matching graph before decoding.

For each dangerous hook edge $G_i$, use a weight based on

$P(H_i \mid F, \text{local syndrome}).$

At minimum, distinguish

$P(H_i \mid F_i=1)$

from

$P(H_i \mid F_i=0).$

This gives a decoder of the form

**measured flags $\rightarrow$ conditional edge weights $\rightarrow$ MWPM.**

This may already fix the full-flag circuit if the main problem with correlated matching is that its correlations are activated only through the first-pass MWPM hypothesis.

However, for partial flags with several competing explanations, simple reweighting may still double-count one flag's evidence. The iterative refinement scheme is intended to resolve this ambiguity explicitly.

---

# 12. Interpretation as a sparse hypergraph decoder

The full maximum-likelihood problem can be written in terms of physical fault variables $x_i$:

minimize

$\sum_i w_i x_i$

subject to

$\bigoplus_i D_i x_i = s.$

A general hypergraph decoder attempts to solve this problem directly.

The special structure here is that almost all faults are already graphlike. Only a sparse set of interface faults require non-graphlike treatment.

The proposed decoder therefore decomposes the problem into:

- a **small local inference problem** over flagged hyperedges;
- a **large graphlike residual problem** handled by MWPM.

In this sense it approximates hypergraph decoding without paying the cost of a fully general hypergraph solver.

---

# 13. Relation to ghost-edge decoding

The proposal is closely related to iterative ghost-edge decoding developed for structured hyperedges in fault-tolerant circuits.

The generic ghost-edge idea is:

1. split a physical hyperedge into graphlike fragments;
2. let MWPM identify a fragment;
3. interpret selection of the fragment as evidence for the full hyperedge;
4. XOR the complete hyperedge from the syndrome;
5. repeat;
6. perform a final matching pass without artificial shortcut edges.

The Hadamard flag problem has additional structure:

> a measured flag already identifies a very small set of possible hyperedges.

Therefore the ghost hypotheses can be **gated by active flags** rather than being globally available.

This should reduce both ambiguity and computational overhead.

---

# 14. Relation to existing flag decoding

The general idea of using flag information during decoding is not new.

Relevant approaches include:

### Deterministic deflagging

A flag pattern directly triggers a virtual Pauli correction. Subsequent syndrome information is adjusted accordingly.

This works particularly well when a given flag pattern has an essentially unique dangerous interpretation.

### Flag-aware MWPM / conditional weights

Flag outcomes are used to modify matching weights so that hook errors associated with triggered flags become more likely.

### Lookup-table flag decoding

The joint syndrome and flag vector are used to identify an appropriate correction.

This can preserve distance but does not naturally scale to large spacetime circuits.

### Correlated matching

Correlations between graph components arising from the same physical mechanism are approximately included through conditional reweighting.

### General hypergraph decoders

The full detector error model is decoded without reducing it to an ordinary graph. This retains the correlations but can be much more computationally expensive.

The proposed decoder lies between conditional-weight matching and full hypergraph decoding.

Its distinctive ingredient is:

> For an ambiguous flag, use the global syndrome and MWPM to choose a concrete physical hyperedge, commit that hyperedge, remove its complete detector signature, and continue decoding the residual syndrome.

This should be viewed as a specialized approximation to the same underlying hypergraph inference problem, not as a fundamentally unrelated decoding paradigm.

---

# 15. Why this may restore the full distance

The decoder should preserve the following physical distinction.

A flagged dangerous hook is a one-fault event:

$P(H_i) = O(p).$

An unflagged occurrence of the same dangerous hook pattern generally requires an additional failure that hides or alters the flag:

$P(G_i \text{ without expected } F_i) = O(p^2)$

in the relevant cases.

A decoder that places $G_i$ into the final graph as a generic $O(p)$ shortcut loses this distinction and can reduce the effective distance.

The proposed decoder instead allows the cheap shortcut only as a temporary hypothesis associated with the correct flag evidence.

Once the flagged hyperedge has been inferred, the entire event is removed.

If the expected flag is absent, the decoder should not have access to the same cheap shortcut.

This reproduces the physical fault counting required for full-distance decoding.

---

# 16. Expected computational cost

Let $M$ denote the cost of one MWPM call.

A general hypergraph decoder may be orders of magnitude slower than matching.

The proposed decoder should require approximately

$O(rM)$

work per shot, where $r$ is the number of refinement passes.

Because dangerous correlations are restricted to a thin Hadamard interface and flags are sparse at low physical error rate, one expects $r$ to remain small.

A practical target is:

- 1 final MWPM call;
- 1-3 proposal/refinement calls in typical shots.

Thus the decoder could potentially retain most of the speed advantage of MWPM while recovering the distance obtained by a more general hypergraph decoder.

---

# 17. Suggested implementation stages

## Stage 1: understand correlated matching failure

For individual injected physical faults, inspect:

- the complete DEM mechanism;
- its decomposition into graphlike pieces;
- first-pass PyMatching correction;
- which correlations are activated;
- second-pass reweighted graph;
- final correction.

Do this separately for:

1. left hooks with full flags;
2. right hooks with full flags;
3. left hooks with partial flags;
4. delayed right hooks with partial flags.

The full-flag case is especially important. If each dangerous hook has an essentially unique flag and correlated matching still loses distance, understanding the exact failure mechanism should guide the simplest possible fix.

## Stage 2: direct flag-conditioned reweighting

Use measured flags directly to set hook-edge weights before the first MWPM pass.

Test whether this restores full distance for the full-flag circuit.

## Stage 3: flag-gated iterative refinement

For partial flags:

1. generate all candidate hook mechanisms associated with each active flag;
2. expose their graphlike components as ghost hypotheses;
3. run MWPM;
4. commit selected full hyperedges;
5. update the syndrome;
6. iterate;
7. run final ordinary MWPM.

## Stage 4: clustering

If several nearby flags interact, build connected clusters using a bipartite graph

**flags $\leftrightarrow$ candidate physical faults.**

Decode independent clusters separately.

At low $p$, most clusters should be very small.

## Stage 5: beam search if needed

If greedy commitment causes failures, retain the best $K$ flag interpretations.

For each hypothesis:

1. remove the hypothesized hyperedges;
2. decode the residual syndrome using MWPM;
3. score

$W = W_{\rm flagged\ hypotheses} + W_{\rm residual\ MWPM};$

4. keep the best few states.

A small beam width may approximate maximum likelihood much better while remaining fast.

---

# 18. Key experiments

The main comparison should include:

- standard PyMatching;
- correlated PyMatching;
- direct flag-conditioned MWPM;
- iterative flag-gated syndrome refinement;
- Tesseract or another general decoder as a reference.

For each decoder measure:

### Logical error scaling

Fit the effective distance and test whether

$d_{\rm eff}=2k+1$

is recovered.

### Runtime

Measure decoding time per shot as a function of distance and physical error rate.

### Number of refinement passes

Determine how often additional iterations are required.

### Failure mechanism

At low $p$, enumerate minimum-order logical failures and determine whether any decoder-induced artificial shortcut remains.

---

# 19. Most important conceptual test

The key question is not merely whether the new decoder has lower logical error rate.

It is:

> Does every logical failure produced by the decoder require the correct minimum number of underlying physical faults?

If yes, then the decoder preserves the circuit distance.

This can be checked directly at small distance by enumerating low-weight physical fault sets and verifying that no set below the target fault distance causes a logical failure.

---

# 20. Possible outcome

There are three plausible outcomes.

### Outcome A: direct flag-conditioned weighting is enough

Then the main issue with correlated matching is that correlation activation is mediated through the first MWPM hypothesis rather than directly through measured flag outcomes.

The simplest decoder would be preferable.

### Outcome B: direct conditioning fixes full flags but not partial flags

This would support the hypothesis that **ambiguity between several physical explanations of the same flag** is the essential difficulty.

Iterative commitment / syndrome refinement would then have a clear purpose.

### Outcome C: iterative refinement also fails

Then the relevant correlations are more global than expected, or greedy hard decisions discard too much posterior information.

The next step would be a local beam search or BP-assisted version rather than immediately moving to a fully general hypergraph decoder.

---

# 21. Summary

The proposed decoder exploits a special property of the spatial Hadamard circuit:

> the non-graphlike faults that matter are sparse, local, and explicitly heralded by flag measurements.

Instead of asking a general hypergraph decoder to solve the entire circuit, use the flags to isolate a small family of candidate physical faults.

The proposed pipeline is

**flag syndrome  
$\rightarrow$ candidate flagged hyperedges  
$\rightarrow$ MWPM resolves ambiguity using global syndrome  
$\rightarrow$ commit complete physical fault  
$\rightarrow$ refine syndrome  
$\rightarrow$ repeat  
$\rightarrow$ final ordinary MWPM.**

This is closely related to correlated matching and ghost-edge decoding, but differs from standard correlated matching in that it can make an explicit physical-fault decision and remove the entire hyperedge from the syndrome.

The immediate research question is whether this distinction is sufficient to restore

$d_{\rm eff}=2k+1$

for the flagged spatial Hadamard circuit while requiring only a small constant number of matching calls per shot.
