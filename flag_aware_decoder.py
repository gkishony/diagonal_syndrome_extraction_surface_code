#!/usr/bin/env python3
"""Flag-aware syndrome-conditioned matching for circuits with flagged hyperedges.

The spatial Hadamard construction of arXiv:2602.09099 measures flag qubits that
herald the dangerous hook errors of the stretched stabilizers.  Such a mechanism
has the detector signature

    H_i = G_i  xor  F_i

with ``F_i`` the flag detectors and ``G_i`` the ordinary detectors.  Stim
decomposes it into graphlike pieces, some of which contain the flag detectors
and some of which do not.  A matching decoder treats those pieces as independent
errors, so the flag-free part of a hook stays available as a weight-``p``
shortcut even in shots where the flags stayed quiet.  That is what costs the
decoder its effective distance.

``FlagAwareMatching`` starts from the same matching graph an ordinary decoder
would build, flag detectors included, and then makes the flag-free pieces of the
flagged mechanisms *conditional on the measured flags*:

* if the flags of a mechanism did not all fire, its flag-free pieces are priced
  as the mechanism plus the extra fault that would have to silence the flags,
  which restores the two-fault cost of an unheralded hook;
* if they did fire, the pieces are restored to the bare single-fault weight, and
  optionally the whole mechanism is re-priced so that using all of its pieces
  together costs one fault rather than one per piece.

Two strategies are built on that graph:

``mode="reweight"``
    A single matching call on the gated graph.

``mode="iterative"``
    The gated graph is a *proposal* mechanism.  Mechanisms whose flag-free pieces
    matching selected become candidate hypotheses; each is scored by the weight
    of the mechanism plus the weight of re-matching the residual syndrome with
    that mechanism's evidence removed.  The best hypothesis is committed only if
    it beats the un-committed solution: its detector signature is XOR-ed out of
    the working syndrome, its logical effect is accumulated, and its flags are
    consumed so competing mechanisms cannot reuse the same evidence.  Scoring the
    hypotheses instead of taking the first plausible one matters: committing
    greedily costs more distance than it recovers.

``mode="hyperedge"``
    The same scored commitment, but the candidates are every mechanism stim had
    to split across several edges, not only the flagged ones.  Flags are still
    used to re-price the hooks they herald.  This covers the hyperedges that
    carry no flag at all, which is where flag conditioning alone runs out.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pymatching
import stim

# Edge keys are ``(node, None)`` for boundary edges and ``(low, high)`` otherwise.
EdgeKey = tuple[int, Optional[int]]

# Probability standing in for "cannot happen"; pymatching needs a finite weight
# and this corresponds to a weight of ~34, far above any relevant matching path.
_FORBIDDEN_PROBABILITY = 1e-15

Component = tuple[frozenset[int], frozenset[int]]

_MODES = ("reweight", "iterative", "hyperedge")


def _weight(probability: float) -> float:
    """Return the log-likelihood weight of an error with the given probability."""
    probability = min(max(probability, _FORBIDDEN_PROBABILITY), 1.0 - 1e-15)
    return math.log((1.0 - probability) / probability)


def _xor_merge(probabilities: Iterable[float]) -> float:
    """Combine independent error probabilities acting on the same edge."""
    total = 0.0
    for p in probabilities:
        total = total * (1.0 - p) + p * (1.0 - total)
    return total


def _edge_key(detectors: Iterable[int]) -> EdgeKey:
    ordered = sorted(detectors)
    if len(ordered) == 1:
        return (ordered[0], None)
    return (ordered[0], ordered[1])


def flag_detectors_from_dem(dem: stim.DetectorErrorModel, tag: str = "flag") -> set[int]:
    """Return the detectors that the circuit marked as flag measurements.

    The spatial Hadamard construction tags its flag detectors, and stim carries
    the tag through to the detector error model, so a decoder handed nothing but
    a DEM can still tell which detectors are flags.
    """
    marker = f"[{tag}]"
    flags: set[int] = set()
    for instruction in dem.flattened():
        if instruction.type != "detector" or marker not in str(instruction):
            continue
        flags.update(
            target.val
            for target in instruction.targets_copy()
            if target.is_relative_detector_id()
        )
    return flags


def _parse_components(instruction: stim.DemInstruction) -> list[Component]:
    """Split a decomposed DEM error into its ``^``-separated components."""
    components: list[Component] = []
    detectors: set[int] = set()
    observables: set[int] = set()
    for target in instruction.targets_copy():
        if target.is_separator():
            components.append((frozenset(detectors), frozenset(observables)))
            detectors, observables = set(), set()
        elif target.is_relative_detector_id():
            detectors.add(target.val)
        elif target.is_logical_observable_id():
            observables.add(target.val)
    components.append((frozenset(detectors), frozenset(observables)))
    return components


@dataclass(frozen=True)
class FlaggedMechanism:
    """A fault mechanism that fires flag detectors as well as ordinary ones."""

    probability: float
    silenced_probability: float
    flags: frozenset[int]
    detectors: frozenset[int]
    observables: frozenset[int]
    # Graphlike pieces of the mechanism that contain no flag detector; these are
    # the ones that can act as an unheralded shortcut.
    open_pieces: tuple[EdgeKey, ...]
    # Every graphlike piece, flag-carrying ones included.
    all_pieces: tuple[EdgeKey, ...]


@dataclass(frozen=True)
class Hypothesis:
    """A mechanism stim had to split across several matching edges.

    Matching prices each piece separately, so a single fault that lands on
    several pieces looks as expensive as several faults.  Re-proposing the whole
    mechanism as one unit is what puts that cost back to one fault.
    """

    probability: float
    detectors: frozenset[int]
    observables: frozenset[int]
    pieces: tuple[EdgeKey, ...]


@dataclass
class _EdgeContributions:
    """Every mechanism mapping onto one edge of the matching graph."""

    static: list[float] = field(default_factory=list)
    gated: list[tuple[int, float]] = field(default_factory=list)
    fault_ids: frozenset[int] = frozenset()
    dominant: float = 0.0
    default_probability: float = 0.0

    def add_static(self, probability: float, fault_ids: frozenset[int]) -> None:
        self.static.append(probability)
        self._adopt(probability, fault_ids)

    def add_gated(self, mechanism: int, silenced: float, fault_ids: frozenset[int]) -> None:
        self.gated.append((mechanism, silenced))
        self._adopt(silenced, fault_ids)

    def _adopt(self, probability: float, fault_ids: frozenset[int]) -> None:
        # pymatching stores one logical mask per edge; keep the dominant one.
        if probability > self.dominant:
            self.dominant = probability
            self.fault_ids = fault_ids

    def finalise(self) -> None:
        self.default_probability = _xor_merge(
            self.static + [silenced for _, silenced in self.gated]
        )

    def gated_probability(self, heralded: dict[int, float]) -> float:
        probabilities = list(self.static)
        for mechanism, silenced in self.gated:
            probabilities.append(heralded.get(mechanism, silenced))
        return _xor_merge(probabilities)


class FlagAwareMatching:
    """Matching decoder that conditions flagged hyperedges on the measured flags.

    Args:
        dem: detector error model decomposed into graphlike components
            (``decompose_errors=True``).
        flag_detectors: indices of the detectors carrying flag measurements.  When
            omitted they are read from the ``flag``-tagged detectors of the DEM.
        mode: ``"reweight"`` for one gated matching call, ``"iterative"`` for
            flag-gated hypothesis commitment followed by a clean pass, or
            ``"hyperedge"`` to let every split mechanism be a candidate.
        max_iterations: cap on the number of commitment rounds per shot.
        max_candidates: hypotheses scored per commitment round in
            ``mode="hyperedge"``; scoring one costs a matching call.
        commit_margin: weight a hypothesis must save before it is committed.
        enable_correlations: use pymatching's two-pass correlated matching on top
            of the flag conditioning.
        share_heralded_weight: when a mechanism is heralded, spread its weight
            over its pieces so that using all of them costs a single fault
            instead of one fault per piece.  Redundant when ``enable_correlations``
            already accounts for the decomposition.
        flag_silencing_probability: fallback probability that a flag pattern stays
            silent, used when the model has no explicit mechanism doing so.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        flag_detectors: Optional[Iterable[int]] = None,
        *,
        mode: str = "reweight",
        max_iterations: int = 3,
        commit_margin: float = 0.0,
        enable_correlations: bool = True,
        share_heralded_weight: bool = False,
        flag_silencing_probability: Optional[float] = None,
        max_candidates: int = 8,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        self.mode = mode
        self.max_iterations = max_iterations
        self.commit_margin = commit_margin
        self.max_candidates = max_candidates
        self.enable_correlations = enable_correlations
        self.share_heralded_weight = share_heralded_weight
        self._dem = dem
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.flag_detectors = frozenset(
            flag_detectors_from_dem(dem) if flag_detectors is None else flag_detectors
        )
        self._silencing_override = flag_silencing_probability

        flat = dem.flattened()
        self._silencing = self._collect_flag_silencing(flat)

        self.mechanisms: list[FlaggedMechanism] = []
        self.hypotheses: list[Hypothesis] = []
        self._edges: dict[EdgeKey, _EdgeContributions] = defaultdict(_EdgeContributions)
        self.dropped_components = 0
        self._build(flat)
        for contributions in self._edges.values():
            contributions.finalise()

        # Mechanisms are indexed by their lowest flag so a shot only scans the
        # candidates attached to the flags that actually fired.
        self._by_flag: dict[int, list[int]] = defaultdict(list)
        for index, mechanism in enumerate(self.mechanisms):
            self._by_flag[min(mechanism.flags)].append(index)

        # Split mechanisms are indexed by piece so a shot only scans the
        # hypotheses that touch the edges matching actually chose, and by
        # detector so it can also reach the ones matching did not choose.
        self._by_piece: dict[EdgeKey, list[int]] = defaultdict(list)
        self._by_detector: dict[int, list[int]] = defaultdict(list)
        for index, hypothesis in enumerate(self.hypotheses):
            for key in hypothesis.pieces:
                self._by_piece[key].append(index)
            for detector in hypothesis.detectors:
                self._by_detector[detector].append(index)

        self._matching = self._build_matching()
        self._flag_mask = np.zeros(self.num_detectors, dtype=bool)
        if self.flag_detectors:
            self._flag_mask[sorted(self.flag_detectors)] = True

        # Diagnostics collected while decoding.
        self.total_commits = 0
        self.total_matching_calls = 0
        self.shots_with_active_flags = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _collect_flag_silencing(self, flat: stim.DetectorErrorModel) -> dict[frozenset[int], float]:
        """Probability of each mechanism that flips flag detectors only.

        A dangerous mechanism occurring without setting off its flags needs an
        extra fault flipping exactly those flag detectors, so these probabilities
        set the price of an unheralded hook.
        """
        silencing: dict[frozenset[int], float] = {}
        for instruction in flat:
            if instruction.type != "error":
                continue
            detectors = frozenset(
                t.val for t in instruction.targets_copy() if t.is_relative_detector_id()
            )
            if not detectors or not detectors <= self.flag_detectors:
                continue
            probability = instruction.args_copy()[0]
            silencing[detectors] = _xor_merge([silencing.get(detectors, 0.0), probability])
        return silencing

    def _silencing_probability(self, flags: frozenset[int]) -> float:
        direct = self._silencing.get(flags)
        if direct is not None:
            return direct
        if self._silencing_override is not None:
            return self._silencing_override
        # No single fault silences this pattern: charge the product of the
        # individual flag flips.
        product = 1.0
        for flag in flags:
            single = self._silencing.get(frozenset({flag}))
            if single is None:
                return _FORBIDDEN_PROBABILITY
            product *= single
        return max(product, _FORBIDDEN_PROBABILITY)

    def _build(self, flat: stim.DetectorErrorModel) -> None:
        for instruction in flat:
            if instruction.type != "error":
                continue
            probability = instruction.args_copy()[0]
            if probability <= 0.0:
                continue
            components = [
                (detectors, observables)
                for detectors, observables in _parse_components(instruction)
                if detectors
            ]
            if not components:
                continue
            if any(len(detectors) > 2 for detectors, _ in components):
                # Stim could not fully decompose this mechanism; matching cannot
                # represent it, exactly as for an ordinary matching decoder.
                self.dropped_components += 1
                continue
            detector_set = frozenset().union(*(detectors for detectors, _ in components))
            if len(components) > 1:
                observable_set: frozenset[int] = frozenset()
                for _, component_observables in components:
                    observable_set ^= component_observables
                self.hypotheses.append(
                    Hypothesis(
                        probability=probability,
                        detectors=detector_set,
                        observables=observable_set,
                        pieces=tuple(_edge_key(d) for d, _ in components),
                    )
                )
            flags = detector_set & self.flag_detectors
            open_pieces = [
                _edge_key(detectors)
                for detectors, _ in components
                if not (detectors & self.flag_detectors)
            ]
            if not flags or not open_pieces:
                # Nothing to condition on: either no flag is involved, or every
                # piece already carries a flag detector and matching must explain
                # it on its own.
                for detectors, observables in components:
                    self._edges[_edge_key(detectors)].add_static(probability, observables)
                continue
            self._add_flagged(components, flags, probability, open_pieces)

    def _add_flagged(
        self,
        components: Sequence[Component],
        flags: frozenset[int],
        probability: float,
        open_pieces: Sequence[EdgeKey],
    ) -> None:
        silenced = probability * self._silencing_probability(flags)
        index = len(self.mechanisms)
        detector_set: set[int] = set()
        observables: set[int] = set()
        all_pieces: list[EdgeKey] = []
        for component_detectors, component_observables in components:
            detector_set |= component_detectors
            observables ^= set(component_observables)
            key = _edge_key(component_detectors)
            all_pieces.append(key)
            if component_detectors & self.flag_detectors:
                # Pieces touching a flag are already constrained by the flag node.
                self._edges[key].add_static(probability, component_observables)
            else:
                self._edges[key].add_gated(index, silenced, component_observables)
        self.mechanisms.append(
            FlaggedMechanism(
                probability=probability,
                silenced_probability=silenced,
                flags=flags,
                detectors=frozenset(detector_set),
                observables=frozenset(observables),
                open_pieces=tuple(open_pieces),
                all_pieces=tuple(all_pieces),
            )
        )

    def _build_matching(self) -> pymatching.Matching:
        """Start from pymatching's own graph and only re-price the gated edges.

        Letting pymatching build the graph keeps its logical masks and, when
        ``enable_correlations`` is set, the decomposition metadata its two-pass
        decoder relies on; rewriting an edge weight in place leaves both intact.
        """
        matching = pymatching.Matching.from_detector_error_model(
            self._dem, enable_correlations=self.enable_correlations
        )
        self._fault_ids: dict[EdgeKey, set[int]] = {}
        self._edge_weight: dict[EdgeKey, float] = {
            (node_a, node_b): data["weight"] for node_a, node_b, data in matching.edges()
        }
        for key, contributions in self._edges.items():
            if not contributions.gated:
                continue
            node_a, node_b = key
            data = (
                matching.get_boundary_edge_data(node_a)
                if node_b is None
                else matching.get_edge_data(node_a, node_b)
            )
            self._fault_ids[key] = set(data["fault_ids"])
            self._write_edge(matching, key, contributions.default_probability)
        return matching

    def _write_edge(self, matching: pymatching.Matching, key: EdgeKey, probability: float) -> None:
        node_a, node_b = key
        weight = _weight(probability)
        self._edge_weight[key] = weight
        fault_ids = self._fault_ids[key]
        if node_b is None:
            matching.add_boundary_edge(
                node_a, fault_ids=fault_ids, weight=weight, merge_strategy="replace"
            )
        else:
            matching.add_edge(
                node_a, node_b, fault_ids=fault_ids, weight=weight, merge_strategy="replace"
            )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _heralded_mechanisms(
        self, active_flags: set[int], committed: set[int]
    ) -> dict[int, float]:
        """Mechanisms whose flags all fired, mapped to the probability to use."""
        heralded: dict[int, float] = {}
        for flag in active_flags:
            for index in self._by_flag.get(flag, ()):
                if index in committed or index in heralded:
                    continue
                mechanism = self.mechanisms[index]
                if not mechanism.flags <= active_flags:
                    continue
                pieces = len(mechanism.all_pieces)
                if self.share_heralded_weight and pieces > 1:
                    # Spread the mechanism over its pieces so that selecting all
                    # of them costs one fault rather than one per piece.
                    heralded[index] = mechanism.probability ** (1.0 / pieces)
                else:
                    heralded[index] = mechanism.probability
        return heralded

    def _apply_gating(
        self, heralded: dict[int, float], previously_gated: set[EdgeKey]
    ) -> set[EdgeKey]:
        gated: set[EdgeKey] = set()
        for index in heralded:
            gated.update(self.mechanisms[index].open_pieces)
        for key in previously_gated - gated:
            self._write_edge(self._matching, key, self._edges[key].default_probability)
        for key in gated:
            self._write_edge(self._matching, key, self._edges[key].gated_probability(heralded))
        return gated

    def _reset_gating(self, gated: Iterable[EdgeKey]) -> None:
        for key in gated:
            self._write_edge(self._matching, key, self._edges[key].default_probability)

    def _decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        self.total_matching_calls += 1
        return self._matching.decode(
            syndrome, enable_correlations=self.enable_correlations
        ).astype(bool)

    def _solve(self, syndrome: np.ndarray) -> tuple[set[EdgeKey], float]:
        """Match the syndrome and return the edges used and their total weight."""
        self.total_matching_calls += 1
        edges = self._matching.decode_to_edges_array(
            syndrome, enable_correlations=self.enable_correlations
        )
        selected: set[EdgeKey] = set()
        for node_a, node_b in edges:
            if node_a == -1:
                selected.add((int(node_b), None))
            elif node_b == -1:
                selected.add((int(node_a), None))
            else:
                selected.add((int(min(node_a, node_b)), int(max(node_a, node_b))))
        return selected, sum(self._edge_weight.get(key, 0.0) for key in selected)

    def _corrected_cost(self, selected: set[EdgeKey]) -> float:
        """Weight of a solution with each split mechanism charged only once.

        Summing edge weights bills a single fault once per piece, which is what
        makes a true explanation look more expensive than a false one.  Wherever
        every piece of a mechanism is present, its pieces are replaced by the
        mechanism itself.  Each edge is discounted at most once, so overlapping
        mechanisms cannot be double-counted.
        """
        total = sum(self._edge_weight.get(key, 0.0) for key in selected)
        spent: set[EdgeKey] = set()
        seen: set[int] = set()
        for key in selected:
            for index in self._by_piece.get(key, ()):
                if index in seen:
                    continue
                seen.add(index)
                hypothesis = self.hypotheses[index]
                pieces = hypothesis.pieces
                if any(p in spent for p in pieces) or not all(p in selected for p in pieces):
                    continue
                pieced = sum(self._edge_weight.get(p, 0.0) for p in pieces)
                discount = pieced - _weight(hypothesis.probability)
                if discount > 0.0:
                    total -= discount
                    spent.update(pieces)
        return total

    def _best_hypothesis(
        self,
        syndrome: np.ndarray,
        heralded: dict[int, float],
        gated: set[EdgeKey],
    ) -> tuple[Optional[int], set[EdgeKey]]:
        """Score the heralded mechanisms matching endorsed and return the best one.

        A hypothesis is worth committing only when the mechanism's own weight plus
        the weight of re-matching the residual syndrome beats leaving the
        syndrome as it is.
        """
        selected, baseline = self._solve(syndrome)
        candidates = [
            index
            for index in heralded
            if self.mechanisms[index].open_pieces
            and selected.issuperset(self.mechanisms[index].open_pieces)
        ]
        best: Optional[int] = None
        best_score = baseline - self.commit_margin
        for index in candidates:
            mechanism = self.mechanisms[index]
            # Take this mechanism's evidence off the table before re-matching, so
            # the residual cost cannot be paid with the same herald twice.
            gated = self._apply_gating(
                {k: v for k, v in heralded.items() if k != index}, gated
            )
            residual = syndrome.copy()
            residual[sorted(mechanism.detectors)] ^= True
            _, cost = self._solve(residual)
            score = _weight(mechanism.probability) + cost
            if score < best_score:
                best_score = score
                best = index
        return best, gated

    def _commit_hyperedges(self, syndrome: np.ndarray) -> np.ndarray:
        """Repeatedly fold the best split mechanism back into one fault.

        Each round asks matching for a solution and scores candidate mechanisms
        by their own weight plus the weight of re-matching the syndrome with
        their detectors removed.  The best is committed when it beats leaving
        the syndrome alone, which is what stops a single fault from being
        charged once per piece.

        Two details decide whether this works.  Both sides of the comparison are
        measured with :meth:`_corrected_cost`, so the baseline is not quietly
        billed more than the candidate.  And candidates are drawn from the lit
        detectors, not only from the edges matching already chose: when a true
        mechanism is overcharged, the solution that hides it usually contains
        none of its pieces, so an edge-keyed search would never propose it.
        """
        prediction = np.zeros(self.num_observables, dtype=bool)
        committed: set[int] = set()
        for _ in range(self.max_iterations):
            selected, _ = self._solve(syndrome)
            baseline = self._corrected_cost(selected)
            lit = set(np.flatnonzero(syndrome).tolist())
            candidates: set[int] = set()
            for key in selected:
                candidates.update(self._by_piece.get(key, ()))
            for detector in lit:
                candidates.update(self._by_detector.get(detector, ()))
            candidates -= committed
            if not candidates:
                break

            # Cheap pre-filter: prefer mechanisms that explain the most lit
            # detectors, then those replacing the most expensive chosen edges.
            def rank(index: int) -> tuple[int, float]:
                hypothesis = self.hypotheses[index]
                covered = sum(
                    self._edge_weight.get(key, 0.0)
                    for key in hypothesis.pieces
                    if key in selected
                )
                explained = len(hypothesis.detectors & lit)
                return (-explained, _weight(hypothesis.probability) - covered)

            ranked = sorted(candidates, key=rank)[: self.max_candidates]

            best: Optional[int] = None
            best_score = baseline - self.commit_margin
            for index in ranked:
                hypothesis = self.hypotheses[index]
                residual = syndrome.copy()
                residual[sorted(hypothesis.detectors)] ^= True
                rest, _ = self._solve(residual)
                score = _weight(hypothesis.probability) + self._corrected_cost(rest)
                if score < best_score:
                    best_score = score
                    best = index
            if best is None:
                break

            hypothesis = self.hypotheses[best]
            committed.add(best)
            syndrome[sorted(hypothesis.detectors)] ^= True
            for fault in hypothesis.observables:
                prediction[fault] ^= True
            self.total_commits += 1

        return prediction ^ self._decode_syndrome(syndrome)

    def decode(self, detectors: np.ndarray) -> np.ndarray:
        """Decode a single shot given the full detector vector."""
        syndrome = np.asarray(detectors, dtype=bool).copy()
        active_flags = set(np.flatnonzero(syndrome & self._flag_mask).tolist())

        if self.mode == "hyperedge":
            gated: set[EdgeKey] = set()
            try:
                if active_flags:
                    self.shots_with_active_flags += 1
                    gated = self._apply_gating(
                        self._heralded_mechanisms(active_flags, set()), gated
                    )
                return self._commit_hyperedges(syndrome)
            finally:
                self._reset_gating(gated)

        if not active_flags:
            return self._decode_syndrome(syndrome)

        self.shots_with_active_flags += 1
        prediction = np.zeros(self.num_observables, dtype=bool)
        committed: set[int] = set()
        gated: set[EdgeKey] = set()
        try:
            if self.mode == "reweight":
                heralded = self._heralded_mechanisms(active_flags, committed)
                gated = self._apply_gating(heralded, gated)
                return self._decode_syndrome(syndrome)

            for _ in range(self.max_iterations):
                heralded = self._heralded_mechanisms(active_flags, committed)
                if not heralded:
                    break
                gated = self._apply_gating(heralded, gated)
                choice, gated = self._best_hypothesis(syndrome, heralded, gated)
                if choice is None:
                    break
                mechanism = self.mechanisms[choice]
                committed.add(choice)
                active_flags -= mechanism.flags
                syndrome[sorted(mechanism.detectors)] ^= True
                for fault in mechanism.observables:
                    prediction[fault] ^= True
                self.total_commits += 1

            self._reset_gating(gated)
            gated = set()
            prediction ^= self._decode_syndrome(syndrome)
            return prediction
        finally:
            self._reset_gating(gated)

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        """Decode a batch of shots, shape ``(num_shots, num_detectors)``."""
        detectors = np.asarray(detectors, dtype=bool)
        num_shots = detectors.shape[0]
        predictions = np.zeros((num_shots, self.num_observables), dtype=bool)

        if self.mode == "hyperedge":
            # Every shot can contain a split mechanism, flags or not.
            flagged_shots = np.ones(num_shots, dtype=bool)
        elif self.flag_detectors:
            flagged_shots = detectors[:, self._flag_mask].any(axis=1)
        else:
            flagged_shots = np.zeros(num_shots, dtype=bool)

        quiet = np.flatnonzero(~flagged_shots)
        if quiet.size:
            predictions[quiet] = self._matching.decode_batch(
                detectors[quiet], enable_correlations=self.enable_correlations
            ).astype(bool)
            self.total_matching_calls += int(quiet.size)
        for index in np.flatnonzero(flagged_shots):
            predictions[index] = self.decode(detectors[index])
        return predictions

    def summary(self) -> dict[str, float]:
        """Return diagnostics about the model and the shots decoded so far."""
        return {
            "num_flag_detectors": len(self.flag_detectors),
            "num_flagged_mechanisms": len(self.mechanisms),
            "num_edges": len(self._edges),
            "dropped_components": self.dropped_components,
            "shots_with_active_flags": self.shots_with_active_flags,
            "total_commits": self.total_commits,
            "total_matching_calls": self.total_matching_calls,
        }
