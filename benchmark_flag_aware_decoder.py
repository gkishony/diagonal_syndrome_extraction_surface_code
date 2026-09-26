#!/usr/bin/env python3
"""Benchmark the flag-aware decoder on the spatial Hadamard circuit.

Sweeps physical error rate and code distance for a set of decoders, fits the
effective distance from the low-``p`` slope of the logical error rate, and
writes both a CSV and a plot.

Examples::

    ./venv/bin/python benchmark_flag_aware_decoder.py --k-values 1 2 3
    ./venv/bin/python benchmark_flag_aware_decoder.py --noise-mode full --decoders pymatching flag_aware
    ./venv/bin/python benchmark_flag_aware_decoder.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import sinter
import stim

from benchmark_spatial_hadamard import apply_interface_only_noise
from compact_circuit import compact_and_delay_init
from spatial_hadamard_manual_construction import generate_spatial_hadamard_circuit
from tqec import NoiseModel

DEFAULT_CSV = "benchmark_data/flag_aware_decoder.csv"
DEFAULT_PLOT = "figures/flag_aware_decoder.png"


# =============================================================================
# Sinter decoder wrappers
# =============================================================================


class FlagAwareSinterDecoder(sinter.Decoder):
    """Sinter wrapper around :class:`flag_aware_decoder.FlagAwareMatching`."""

    def __init__(
        self,
        *,
        mode: str = "iterative",
        enable_correlations: bool = True,
        share_heralded_weight: bool = True,
        max_iterations: int = 3,
        max_candidates: int = 8,
    ) -> None:
        self.mode = mode
        self.enable_correlations = enable_correlations
        self.share_heralded_weight = share_heralded_weight
        self.max_iterations = max_iterations
        self.max_candidates = max_candidates

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: str,
        dets_b8_in_path: str,
        obs_predictions_b8_out_path: str,
        tmp_dir: str,
    ) -> None:
        from flag_aware_decoder import FlagAwareMatching

        dem = stim.DetectorErrorModel.from_file(dem_path)
        decoder = FlagAwareMatching(
            dem,
            mode=self.mode,
            enable_correlations=self.enable_correlations,
            share_heralded_weight=self.share_heralded_weight,
            max_iterations=self.max_iterations,
            max_candidates=self.max_candidates,
        )
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=num_dets, num_observables=0
        )
        stim.write_shot_data_file(
            data=decoder.decode_batch(dets),
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )


class CorrelatedPymatchingDecoder(sinter.Decoder):
    """Sinter wrapper for pymatching's two-pass correlated matching."""

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: str,
        dets_b8_in_path: str,
        obs_predictions_b8_out_path: str,
        tmp_dir: str,
    ) -> None:
        import pymatching

        dem = stim.DetectorErrorModel.from_file(dem_path)
        matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=num_dets, num_observables=0
        )
        stim.write_shot_data_file(
            data=matcher.decode_batch(dets, enable_correlations=True),
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )


class TesseractSinterDecoder(sinter.Decoder):
    """Sinter wrapper for the Tesseract hypergraph decoder."""

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: str,
        dets_b8_in_path: str,
        obs_predictions_b8_out_path: str,
        tmp_dir: str,
    ) -> None:
        import tesseract_decoder.tesseract as tesseract

        dem = stim.DetectorErrorModel.from_file(dem_path)
        config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=200_000,
            det_beam=15,
            beam_climbing=True,
            det_orders=[],
            no_revisit_dets=True,
        )
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=num_dets, num_observables=0
        )
        stim.write_shot_data_file(
            data=tesseract.TesseractDecoder(config).decode_batch(dets),
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )


DECODERS: dict[str, Optional[sinter.Decoder]] = {
    # ``None`` means sinter's own built-in decoder of that name.
    "pymatching": None,
    "correlated_pymatching": CorrelatedPymatchingDecoder(),
    "flag_aware_reweight": FlagAwareSinterDecoder(mode="reweight"),
    "flag_aware": FlagAwareSinterDecoder(mode="iterative"),
    "flag_aware_hyperedge": FlagAwareSinterDecoder(mode="hyperedge"),
    "flag_aware_plain": FlagAwareSinterDecoder(mode="iterative", enable_correlations=False),
    "tesseract": TesseractSinterDecoder(),
}

DECODER_STYLE = {
    "pymatching": ("tab:blue", "o", "PyMatching"),
    "correlated_pymatching": ("tab:orange", "s", "Correlated PyMatching"),
    "flag_aware_reweight": ("tab:green", "^", "Flag-aware (reweight)"),
    "flag_aware": ("tab:red", "D", "Flag-aware (iterative)"),
    "flag_aware_hyperedge": ("tab:brown", "*", "Flag-aware (hyperedge commit)"),
    "flag_aware_plain": ("tab:purple", "v", "Flag-aware (no correlations)"),
    "tesseract": ("black", "p", "Tesseract"),
}


# =============================================================================
# Circuits
# =============================================================================


def build_circuit(
    k: int, physical_error_rate: float, axis: str, flag_config: str, noise_mode: str
) -> stim.Circuit:
    """Return the noisy, compacted spatial Hadamard circuit for one data point."""
    clean = compact_and_delay_init(
        generate_spatial_hadamard_circuit(k=k, axis=axis, flag_config=flag_config)
    )
    noise_model = NoiseModel.uniform_depolarizing(physical_error_rate)
    if noise_mode == "interface_only":
        return apply_interface_only_noise(clean, noise_model, k, axis)
    return noise_model.noisy_circuit(clean)


# =============================================================================
# Sweep
# =============================================================================


@dataclass
class Row:
    decoder: str
    k: int
    distance: int
    flag_config: str
    axis: str
    noise_mode: str
    physical_error_rate: float
    shots: int
    errors: int
    logical_error_rate: float
    error_bar: float
    seconds: float


def run_sweep(
    *,
    k_values: Sequence[int],
    physical_error_rates: Sequence[float],
    decoders: Sequence[str],
    axis: str,
    flag_config: str,
    noise_mode: str,
    max_shots: int,
    max_errors: int,
    num_workers: int,
    save_resume_filepath: Optional[str] = None,
) -> list[Row]:
    tasks = []
    for k in k_values:
        for p in physical_error_rates:
            circuit = build_circuit(k, p, axis, flag_config, noise_mode)
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        "k": k,
                        "d": 2 * k + 1,
                        "p": p,
                        "axis": axis,
                        "flag_config": flag_config,
                        "noise_mode": noise_mode,
                    },
                )
            )
            print(
                f"  built k={k} p={p:.5g}: {circuit.num_qubits} qubits, "
                f"{circuit.num_detectors} detectors",
                flush=True,
            )

    custom = {name: DECODERS[name] for name in decoders if DECODERS[name] is not None}
    if save_resume_filepath:
        os.makedirs(os.path.dirname(save_resume_filepath) or ".", exist_ok=True)
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        decoders=list(decoders),
        custom_decoders=custom,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
        save_resume_filepath=save_resume_filepath,
    )
    return rows_from_stats(stats)


def rows_from_stats(stats: Sequence[sinter.TaskStats]) -> list[Row]:
    rows = []
    for stat in stats:
        shots = stat.shots - stat.discards
        errors = stat.errors
        rate = errors / shots if shots else float("nan")
        rows.append(
            Row(
                decoder=stat.decoder,
                k=stat.json_metadata["k"],
                distance=stat.json_metadata["d"],
                flag_config=stat.json_metadata["flag_config"],
                axis=stat.json_metadata["axis"],
                noise_mode=stat.json_metadata["noise_mode"],
                physical_error_rate=stat.json_metadata["p"],
                shots=shots,
                errors=errors,
                logical_error_rate=rate,
                error_bar=math.sqrt(max(errors, 1)) / shots if shots else float("nan"),
                seconds=stat.seconds,
            )
        )
    return rows


def write_csv(rows: Sequence[Row], path: str) -> None:
    if not rows:
        print("No statistics to write.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"Wrote {len(rows)} rows to {path}")


def read_csv(path: str) -> list[Row]:
    with open(path, newline="") as handle:
        return [
            Row(
                decoder=r["decoder"],
                k=int(r["k"]),
                distance=int(r["distance"]),
                flag_config=r["flag_config"],
                axis=r["axis"],
                noise_mode=r["noise_mode"],
                physical_error_rate=float(r["physical_error_rate"]),
                shots=int(r["shots"]),
                errors=int(r["errors"]),
                logical_error_rate=float(r["logical_error_rate"]),
                error_bar=float(r["error_bar"]),
                seconds=float(r["seconds"]),
            )
            for r in csv.DictReader(handle)
        ]


# =============================================================================
# Effective distance
# =============================================================================


def fit_effective_distance(
    points: Sequence[tuple[float, float, int]],
    *,
    min_errors: int = 10,
    num_points: int = 3,
) -> tuple[Optional[float], int]:
    """Fit ``p_logical ~ p^(t+1)`` and return ``d_eff = 2t + 1`` and the fit size.

    ``points`` are ``(physical_error_rate, logical_error_rate, errors)``.  The
    slope comes from a least-squares line through the lowest few physical error
    rates, since that is where the asymptotic scaling shows.  Points with too few
    observed errors are dropped: their logical rate is dominated by shot noise
    and would otherwise swing the slope by several units.
    """
    usable = sorted((p, q) for p, q, errors in points if p > 0 and q > 0 and errors >= min_errors)
    usable = usable[:num_points]
    if len(usable) < 2:
        return None, len(usable)
    xs = np.log([p for p, _ in usable])
    ys = np.log([q for _, q in usable])
    slope = float(np.polyfit(xs, ys, 1)[0])
    return 2 * (slope - 1) + 1, len(usable)


def summarise(rows: Sequence[Row]) -> None:
    by_decoder_k: dict[tuple[str, int], list[tuple[float, float, int]]] = {}
    for row in rows:
        by_decoder_k.setdefault((row.decoder, row.k), []).append(
            (row.physical_error_rate, row.logical_error_rate, row.errors)
        )

    print()
    print("=" * 86)
    print("Effective distance fitted on the lowest physical error rates")
    print("=" * 86)
    print(f"{'decoder':<26}{'k':>3}{'d=2k+1':>9}{'d_eff':>9}{'fit pts':>9}"
          f"{'p_logical(min p)':>20}{'errors':>9}")
    print("-" * 86)
    for (decoder, k), points in sorted(by_decoder_k.items()):
        d_eff, used = fit_effective_distance(points)
        p, lowest, errors = min(points)
        shown = "n/a" if d_eff is None else f"{d_eff:.2f}"
        print(f"{decoder:<26}{k:>3}{2 * k + 1:>9}{shown:>9}{used:>9}"
              f"{lowest:>20.3e}{errors:>9}")
    print("=" * 86)


def plot(rows: Sequence[Row], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decoders = sorted({row.decoder for row in rows})
    k_values = sorted({row.k for row in rows})
    fig, axes = plt.subplots(
        1, len(k_values), figsize=(5.2 * len(k_values), 4.6), squeeze=False, sharey=True
    )

    for column, k in enumerate(k_values):
        ax = axes[0][column]
        for decoder in decoders:
            points = sorted(
                (row.physical_error_rate, row.logical_error_rate, row.error_bar)
                for row in rows
                if row.decoder == decoder and row.k == k
            )
            if not points:
                continue
            colour, marker, label = DECODER_STYLE.get(decoder, ("gray", "x", decoder))
            xs, ys, errs = zip(*points)
            ax.errorbar(
                xs, ys, yerr=errs, color=colour, marker=marker, label=label, capsize=2, lw=1.4
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate $p$")
        ax.set_title(f"$k={k}$  ($d={2 * k + 1}$)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel("logical error rate")
    axes[0][-1].legend(fontsize=8)

    noise_mode = rows[0].noise_mode if rows else ""
    fig.suptitle(f"Spatial Hadamard, {noise_mode.replace('_', ' ')} noise")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"Wrote plot to {path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--physical-error-rates",
        type=float,
        nargs="+",
        default=list(np.logspace(-3, -2, 5)),
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["pymatching", "correlated_pymatching", "flag_aware", "tesseract"],
        choices=sorted(DECODERS),
    )
    parser.add_argument("--axis", default="y", choices=["x", "y"])
    parser.add_argument("--flag-config", default="partial", choices=["none", "partial", "all"])
    parser.add_argument(
        "--noise-mode", default="interface_only", choices=["interface_only", "full"]
    )
    parser.add_argument("--max-shots", type=int, default=200_000)
    parser.add_argument("--max-errors", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--plot", default=DEFAULT_PLOT)
    parser.add_argument("--plot-only", action="store_true", help="re-plot an existing CSV")
    parser.add_argument(
        "--resume-csv",
        default=None,
        help="sinter checkpoint file; defaults to <csv>.sinter.csv. Shots already "
        "recorded there are reused, so an interrupted sweep can be restarted.",
    )
    args = parser.parse_args()
    if args.resume_csv is None:
        args.resume_csv = args.csv + ".sinter.csv"
    return args


def main() -> None:
    args = parse_args()
    if args.plot_only:
        if os.path.exists(args.csv):
            rows = read_csv(args.csv)
        else:
            rows = rows_from_stats(sinter.read_stats_from_csv_files(args.resume_csv))
            write_csv(rows, args.csv)
    else:
        print("Building circuits...")
        rows = run_sweep(
            k_values=args.k_values,
            physical_error_rates=args.physical_error_rates,
            decoders=args.decoders,
            axis=args.axis,
            flag_config=args.flag_config,
            noise_mode=args.noise_mode,
            max_shots=args.max_shots,
            max_errors=args.max_errors,
            num_workers=args.num_workers,
            save_resume_filepath=args.resume_csv,
        )
        write_csv(rows, args.csv)
    summarise(rows)
    plot(rows, args.plot)


if __name__ == "__main__":
    main()
