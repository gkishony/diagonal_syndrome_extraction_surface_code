#!/usr/bin/env python3
"""Measure how many physical faults each decoder needs before it fails.

This is the conceptual test of the flag-aware decoder proposal: a decoder
preserves the circuit distance only if every logical failure it produces
requires the correct minimum number of underlying faults.  Rather than reading
that off a Monte Carlo slope, this script enumerates fault sets of increasing
weight drawn from the detector error model and reports the smallest weight that
fools each decoder.

A decoder that first fails at weight ``t + 1`` has effective distance
``d_eff = 2t + 1``, which is the quantity the paper fits from simulation data.

Example::

    ./venv/bin/python measure_decoder_fault_distance.py --k 1 --max-weight 2
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pymatching
import stim

from benchmark_flag_aware_decoder import build_circuit
from flag_aware_decoder import FlagAwareMatching, flag_detectors_from_dem


@dataclass(frozen=True)
class Fault:
    detectors: tuple[int, ...]
    observables: tuple[int, ...]


def extract_faults(dem: stim.DetectorErrorModel) -> list[Fault]:
    """One entry per physical fault mechanism in the model."""
    faults = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.add(target.val)
            elif target.is_logical_observable_id():
                observables.add(target.val)
        faults.append(Fault(tuple(sorted(detectors)), tuple(sorted(observables))))
    return faults


class ExplanationIndex:
    """Answers "which logical classes explain this syndrome with <= w faults?".

    A decoder that disagrees with the truth has only really lost distance if no
    fault set of the same weight reproduces the syndrome in the class it
    predicted.  When such a set exists the two explanations are indistinguishable
    and picking either one is legitimate.
    """

    def __init__(self, faults: Sequence[Fault], num_observables: int) -> None:
        self._faults = faults
        self._num_observables = num_observables
        self._singles: dict[frozenset[int], set[tuple[bool, ...]]] = {}
        for fault in faults:
            self._singles.setdefault(frozenset(fault.detectors), set()).add(
                self._to_class(fault.observables)
            )
        self._pairs: Optional[dict[frozenset[int], set[tuple[bool, ...]]]] = None

    def _to_class(self, observables) -> tuple[bool, ...]:
        vector = [False] * self._num_observables
        for observable in observables:
            vector[observable] ^= True
        return tuple(vector)

    def _pair_index(self) -> dict[frozenset[int], set[tuple[bool, ...]]]:
        if self._pairs is None:
            self._pairs = {}
            for a, b in itertools.combinations(self._faults, 2):
                key = frozenset(a.detectors) ^ frozenset(b.detectors)
                self._pairs.setdefault(key, set()).add(
                    self._to_class(a.observables + b.observables)
                )
        return self._pairs

    def classes(self, syndrome: frozenset[int], max_weight: int) -> set[tuple[bool, ...]]:
        """Logical classes reachable with at most ``max_weight`` faults."""
        found: set[tuple[bool, ...]] = set()
        found |= self._singles.get(syndrome, set())
        if max_weight >= 2:
            for fault in self._faults:
                residual = syndrome ^ frozenset(fault.detectors)
                own = self._to_class(fault.observables)
                for other in self._singles.get(residual, ()):
                    found.add(tuple(x ^ y for x, y in zip(own, other)))
        if max_weight >= 3:
            pairs = self._pair_index()
            for fault in self._faults:
                residual = syndrome ^ frozenset(fault.detectors)
                own = self._to_class(fault.observables)
                for other in pairs.get(residual, ()):
                    found.add(tuple(x ^ y for x, y in zip(own, other)))
        if max_weight > 3:
            raise ValueError("explanation search is only implemented up to weight 3")
        return found


def _combinations(
    faults: Sequence[Fault], weight: int, sample: Optional[int], seed: int
) -> Iterator[tuple[int, ...]]:
    """All fault-index combinations of the given weight, or a random sample."""
    if sample is None:
        yield from itertools.combinations(range(len(faults)), weight)
        return
    rng = random.Random(seed)
    indices = range(len(faults))
    for _ in range(sample):
        yield tuple(rng.sample(indices, weight))


def first_failing_weight(
    decode: Callable[[np.ndarray], np.ndarray],
    faults: Sequence[Fault],
    num_detectors: int,
    num_observables: int,
    max_weight: int,
    sample: Optional[int],
    seed: int,
    index: ExplanationIndex,
) -> tuple[Optional[int], Optional[tuple[int, ...]], int, int]:
    """Return the lowest weight that makes the decoder mispredict.

    Two kinds of mismatch are not counted.  Fault sets whose syndrome is empty
    while flipping the observable are undetectable logical errors of the circuit
    itself.  And a mismatch is degenerate, not a distance loss, when some other
    fault set of the same weight produces the same syndrome in the class the
    decoder predicted.
    """
    checked = 0
    degenerate = 0
    for weight in range(1, max_weight + 1):
        syndrome = np.zeros(num_detectors, dtype=bool)
        for combination in _combinations(faults, weight, sample, seed + weight):
            syndrome[:] = False
            truth = np.zeros(num_observables, dtype=bool)
            for i in combination:
                fault = faults[i]
                syndrome[list(fault.detectors)] ^= True
                for observable in fault.observables:
                    truth[observable] ^= True
            checked += 1
            if not syndrome.any():
                continue
            prediction = decode(syndrome)
            if np.array_equal(prediction, truth):
                continue
            lit = frozenset(np.flatnonzero(syndrome).tolist())
            if tuple(bool(x) for x in prediction) in index.classes(lit, weight):
                degenerate += 1
                continue
            return weight, combination, checked, degenerate
    return None, None, checked, degenerate


def build_decoders(dem: stim.DetectorErrorModel, flags: set[int]) -> dict[str, Callable]:
    plain = pymatching.Matching.from_detector_error_model(dem)
    correlated = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    decoders: dict[str, Callable] = {
        "pymatching": lambda s: plain.decode(s).astype(bool),
        "correlated_pymatching": lambda s: correlated.decode(
            s, enable_correlations=True
        ).astype(bool),
    }
    for mode in ("reweight", "iterative", "hyperedge"):
        for correlations in (False, True):
            decoder = FlagAwareMatching(
                dem,
                flags,
                mode=mode,
                enable_correlations=correlations,
                share_heralded_weight=True,
            )
            suffix = "" if correlations else "_nocorr"
            decoders[f"flag_aware_{mode}{suffix}"] = decoder.decode
    return decoders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--axis", default="y", choices=["x", "y"])
    parser.add_argument("--flag-config", default="partial", choices=["none", "partial", "all"])
    parser.add_argument(
        "--noise-mode", default="interface_only", choices=["interface_only", "full"]
    )
    parser.add_argument("--physical-error-rate", type=float, default=0.001)
    parser.add_argument("--max-weight", type=int, default=2)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="check this many random fault sets per weight instead of all of them",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-tesseract", action="store_true")
    parser.add_argument(
        "--decoders", nargs="*", default=None, help="only run these decoders"
    )
    args = parser.parse_args()

    circuit = build_circuit(
        args.k, args.physical_error_rate, args.axis, args.flag_config, args.noise_mode
    )
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    flags = flag_detectors_from_dem(dem)
    faults = extract_faults(dem)

    print(f"k={args.k} d={2 * args.k + 1} axis={args.axis} flags={args.flag_config} "
          f"noise={args.noise_mode}")
    print(f"  {dem.num_detectors} detectors ({len(flags)} flags), {len(faults)} fault mechanisms")
    print(f"  enumerating fault sets up to weight {args.max_weight}"
          + (f", sampling {args.sample} per weight" if args.sample else ""))

    decoders = build_decoders(dem, flags)
    if args.include_tesseract or (args.decoders and "tesseract" in args.decoders):
        import tesseract_decoder.tesseract as tesseract

        config = tesseract.TesseractConfig(
            dem=circuit.detector_error_model(decompose_errors=False),
            pqlimit=200_000,
            det_beam=15,
            beam_climbing=True,
            det_orders=[],
            no_revisit_dets=True,
        )
        engine = tesseract.TesseractDecoder(config)
        decoders["tesseract"] = lambda s: engine.decode(s).astype(bool)

    if args.decoders:
        decoders = {name: decoders[name] for name in args.decoders}

    index = ExplanationIndex(faults, dem.num_observables)

    print()
    print(f"{'decoder':<28}{'first failing weight':>22}{'d_eff':>8}{'checked':>12}"
          f"{'ties':>8}{'time':>9}")
    print("-" * 87)
    for name, decode in decoders.items():
        start = time.time()
        weight, example, checked, degenerate = first_failing_weight(
            decode,
            faults,
            dem.num_detectors,
            dem.num_observables,
            args.max_weight,
            args.sample,
            args.seed,
            index,
        )
        elapsed = time.time() - start
        if weight is None:
            shown, d_eff = f"> {args.max_weight}", f"> {2 * args.max_weight - 1}"
        else:
            shown, d_eff = str(weight), str(2 * weight - 1)
        print(f"{name:<28}{shown:>22}{d_eff:>8}{checked:>12}{degenerate:>8}{elapsed:>8.1f}s")
        if example is not None:
            detail = "; ".join(
                f"D{list(faults[i].detectors)}L{list(faults[i].observables)}" for i in example
            )
            print(f"{'':<28}example: {detail}")


if __name__ == "__main__":
    main()
