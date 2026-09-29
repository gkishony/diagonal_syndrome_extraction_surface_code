#!/usr/bin/env python3
"""Which part of the flag conditioning actually buys the distance?

``FlagAwareMatching`` does two separate things to a flagged hyperedge:

* when the flags stayed quiet, its flag-free pieces are priced as the mechanism
  plus the fault that would have to silence the flags (the *silencing penalty*,
  which is the proposal's headline idea);
* when the flags all fired, the mechanism is spread over its pieces so that
  using all of them costs one fault rather than one per piece (*shared weight*).

Turning each off separately says which one matters.  Two metrics are reported:
the number of single faults the decoder mis-corrects, which is what sets the
effective distance, and the failure rate on random fault sets of a larger
weight, which measures how much room is left once the distance is saturated.
Larger ``k`` needs a larger sampled weight to see any failures at all.

Example::

    ./venv/bin/python ablate_flag_aware_decoder.py --k-values 1 2 --flag-configs partial all
"""

from __future__ import annotations

import argparse

import numpy as np

from benchmark_flag_aware_decoder import build_circuit
from flag_aware_decoder import FlagAwareMatching, flag_detectors_from_dem
from measure_decoder_fault_distance import ExplanationIndex, extract_faults

# ``flag_silencing_probability=1.0`` makes an unheralded hook cost exactly the
# mechanism, i.e. removes the penalty without touching anything else.
ABLATIONS = {
    "full decoder": dict(share_heralded_weight=True),
    "shared weight OFF": dict(share_heralded_weight=False),
    "silencing penalty OFF": dict(share_heralded_weight=True, flag_silencing_probability=1.0),
    "both OFF": dict(share_heralded_weight=False, flag_silencing_probability=1.0),
}


def failures(decode, cases, index, weight) -> int:
    """Fault sets the decoder gets genuinely wrong, ignoring legitimate ties."""
    bad = 0
    for syndrome, truth in cases:
        prediction = decode(syndrome)
        if np.array_equal(prediction, truth):
            continue
        lit = frozenset(np.flatnonzero(syndrome).tolist())
        if tuple(bool(x) for x in prediction) in index.classes(lit, weight):
            continue
        bad += 1
    return bad


def make_cases(faults, num_detectors, num_observables, weight, samples, seed):
    """Every single fault, or a random sample of higher-weight fault sets."""
    rng = np.random.default_rng(seed)
    combos = (
        [(i,) for i in range(len(faults))]
        if weight == 1
        else [tuple(rng.integers(0, len(faults), weight)) for _ in range(samples)]
    )
    cases = []
    for combo in combos:
        if len(set(combo)) != len(combo):
            continue
        syndrome = np.zeros(num_detectors, dtype=bool)
        truth = np.zeros(num_observables, dtype=bool)
        for i in combo:
            syndrome[list(faults[i].detectors)] ^= True
            for o in faults[i].observables:
                truth[o] ^= True
        if syndrome.any():
            cases.append((syndrome, truth))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--flag-configs", nargs="+", default=["partial", "all"])
    parser.add_argument("--axis", default="y", choices=["x", "y"])
    parser.add_argument("--noise-mode", default="interface_only")
    parser.add_argument("--physical-error-rate", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--sample-weight", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", default="reweight")
    args = parser.parse_args()

    for k in args.k_values:
        for flag_config in args.flag_configs:
            circuit = build_circuit(
                k, args.physical_error_rate, args.axis, flag_config, args.noise_mode
            )
            dem = circuit.detector_error_model(
                decompose_errors=True, ignore_decomposition_failures=True
            )
            flags = flag_detectors_from_dem(dem)
            faults = extract_faults(dem)
            index = ExplanationIndex(faults, dem.num_observables)

            w = args.sample_weight
            singles = make_cases(faults, dem.num_detectors, dem.num_observables, 1, 0, args.seed)
            sampled = make_cases(
                faults, dem.num_detectors, dem.num_observables, w, args.samples, args.seed
            )

            print(f"\nk={k}  flags={flag_config}  (true distance {2 * k + 1})")
            print(f"   {dem.num_detectors} detectors, {len(flags)} flags, "
                  f"{len(faults)} mechanisms")
            print(f"   {len(singles)} single faults (all of them), "
                  f"{len(sampled)} random weight-{w} fault sets")
            print(f"   {'configuration':<26}{'weight-1 wrong':>16}{f'weight-{w} wrong':>16}"
                  f"{'d_eff':>10}")
            print("   " + "-" * 65)
            for label, kwargs in ABLATIONS.items():
                decoder = FlagAwareMatching(
                    dem, flags, mode=args.mode, enable_correlations=True, **kwargs
                )
                one = failures(decoder.decode, singles, index, 1)
                many = failures(decoder.decode, sampled, index, w)
                d_eff = "1" if one else (f"{2 * w - 1}" if many else f">= {2 * w + 1}")
                print(f"   {label:<26}{one:>16}{many:>16}{d_eff:>10}")


if __name__ == "__main__":
    main()
