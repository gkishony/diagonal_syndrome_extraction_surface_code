#!/usr/bin/env python3
"""Unit tests for the flag-aware matching decoder.

Run with ``./venv/bin/python -m pytest test_flag_aware_decoder.py``.
"""

import numpy as np
import pymatching
import stim

from flag_aware_decoder import FlagAwareMatching, _weight


def _toy_dem() -> stim.DetectorErrorModel:
    """A three-detector chain with one flagged hook error.

    Detectors 0, 1, 2 form a chain to the boundary; detector 3 is a flag.  The
    hook ``D0 D2`` short-circuits the chain, carries the observable, and is
    heralded by the flag.
    """
    return stim.DetectorErrorModel(
        """
        error(0.01) D0 D1
        error(0.01) D1 D2
        error(0.001) D0
        error(0.0005) D2 L0
        error(0.002) D0 D2 L0 ^ D3
        error(0.001) D3
        """
    )


def test_model_separates_flagged_mechanisms():
    decoder = FlagAwareMatching(_toy_dem(), flag_detectors={3}, mode="iterative")
    assert len(decoder.mechanisms) == 1
    mechanism = decoder.mechanisms[0]
    assert mechanism.flags == frozenset({3})
    assert mechanism.detectors == frozenset({0, 2, 3})
    assert mechanism.observables == frozenset({0})
    assert mechanism.open_pieces == ((0, 2),)
    assert set(mechanism.all_pieces) == {(0, 2), (3, None)}
    # The hook is priced as itself plus the fault that silences the flag.
    assert mechanism.silenced_probability == 0.002 * 0.001


def test_graph_matches_pymatching_when_no_detector_is_a_flag():
    """With nothing to condition on the decoder must reduce to plain matching."""
    dem = _toy_dem()
    detectors = np.array(
        [[(bits >> i) & 1 for i in range(4)] for bits in range(16)], dtype=bool
    )
    for correlations in (False, True):
        decoder = FlagAwareMatching(
            dem, flag_detectors=set(), mode="reweight", enable_correlations=correlations
        )
        reference = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=correlations
        ).decode_batch(detectors, enable_correlations=correlations)
        assert np.array_equal(decoder.decode_batch(detectors), reference)


def test_hook_costs_two_faults_without_the_flag_and_one_with_it():
    decoder = FlagAwareMatching(_toy_dem(), flag_detectors={3}, mode="reweight")
    detour = 2 * _weight(decoder._edges[(0, 1)].default_probability)
    assert _weight(decoder._edges[(0, 2)].default_probability) > detour
    assert _weight(decoder._edges[(0, 2)].gated_probability({0: 0.002})) < detour


def test_flagged_hook_is_corrected_and_unflagged_one_is_not():
    decoder = FlagAwareMatching(_toy_dem(), flag_detectors={3}, mode="iterative")

    # Flag fired together with D0 and D2: the hook is the single-fault
    # explanation, so the observable must be flipped.
    flagged = np.zeros(4, dtype=bool)
    flagged[[0, 2, 3]] = True
    assert bool(decoder.decode(flagged)[0]) is True
    assert decoder.total_commits == 1

    # Same ordinary syndrome without the flag: the hook now needs two faults, so
    # the chain D0-D1-D2 is preferred and the observable stays put.
    quiet = np.zeros(4, dtype=bool)
    quiet[[0, 2]] = True
    assert bool(decoder.decode(quiet)[0]) is False


def test_plain_matching_is_fooled_by_the_unflagged_hook():
    """The failure the decoder is meant to fix, for contrast."""
    plain = pymatching.Matching.from_detector_error_model(_toy_dem())
    quiet = np.zeros(4, dtype=bool)
    quiet[[0, 2]] = True
    assert bool(plain.decode(quiet)[0]) is True


def test_reweight_and_iterative_agree_on_the_toy_model():
    dem = _toy_dem()
    iterative = FlagAwareMatching(dem, flag_detectors={3}, mode="iterative")
    reweight = FlagAwareMatching(dem, flag_detectors={3}, mode="reweight")
    for bits in range(16):
        detectors = np.array([(bits >> i) & 1 for i in range(4)], dtype=bool)
        assert np.array_equal(iterative.decode(detectors), reweight.decode(detectors))


def test_batch_matches_single_shot():
    dem = _toy_dem()
    decoder = FlagAwareMatching(dem, flag_detectors={3}, mode="iterative")
    detectors = np.array(
        [[(bits >> i) & 1 for i in range(4)] for bits in range(16)], dtype=bool
    )
    assert np.array_equal(
        decoder.decode_batch(detectors), np.array([decoder.decode(row) for row in detectors])
    )


def _ring_dem(length: int = 12, p: float = 0.02) -> tuple[stim.DetectorErrorModel, set[int]]:
    """A ring of detectors whose flagged hooks halve the matching distance.

    Ordinary errors connect neighbours, hooks connect next-nearest neighbours and
    set a flag.  Ignoring the flags, ``length / 2`` hooks wrap the ring and flip
    the observable, so an unflagged matching decoder sees half the distance.
    """
    lines = []
    for i in range(length):
        observable = " L0" if i == 0 else ""
        lines.append(f"error({p}) D{i} D{(i + 1) % length}{observable}")
    for i in range(length):
        flag = length + i
        observable = " L0" if i >= length - 2 else ""
        lines.append(f"error({p}) D{i} D{(i + 2) % length}{observable} ^ D{flag}")
        lines.append(f"error({p / 10}) D{flag}")
    return stim.DetectorErrorModel("\n".join(lines)), set(range(length, 2 * length))


def test_flag_awareness_beats_plain_matching_on_the_ring():
    dem, flags = _ring_dem()
    detectors, observables, _ = dem.compile_sampler(seed=7).sample(20000)

    plain = pymatching.Matching.from_detector_error_model(dem)
    plain_errors = np.count_nonzero(
        np.any(plain.decode_batch(detectors) != observables, axis=1)
    )

    for mode in ("reweight", "iterative"):
        for share in (False, True):
            decoder = FlagAwareMatching(dem, flags, mode=mode, share_heralded_weight=share)
            errors = np.count_nonzero(
                np.any(decoder.decode_batch(detectors) != observables, axis=1)
            )
            assert errors < plain_errors, f"{mode}/{share}: {errors} vs plain {plain_errors}"
