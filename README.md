# Diagonal Schedule Syndrome Extraction - Surface Code Experiments

This repository contains implementations comparing standard fixed-bulk conventions vs diagonal schedules for surface code syndrome extraction in two scenarios:

## Files

### Core Implementation
- **`diagonal_plaquettes.py`**: Custom `DiagonalPlaquetteGenerator` that provides diagonal schedule plaquettes (schedule [6,4,3,5] for Z and [1,4,3,2] for X). Handles both memory experiments and X-junction spatial cubes.

### Comparison Scripts

#### 1. Memory Experiment Comparison (`complete_comparison.py`)
Compares standard vs diagonal schedule memory experiments:
```bash
# Basic comparison (k=1,2,3, default noise levels)
./venv/bin/python complete_comparison.py

# Custom k values
./venv/bin/python complete_comparison.py --k-values 2 3 4

# Custom shots and noise levels
./venv/bin/python complete_comparison.py --shots 100000 --noise-levels 0.001 0.002 0.003

# Skip certain analyses
./venv/bin/python complete_comparison.py --skip-distance --skip-logical-error
```

#### 2. X-Junction Comparison (`compare_x_junction.py`)
Compares standard vs diagonal schedule for spatial X-junctions:
```bash
# Basic comparison (distance, detector count, etc.)
./venv/bin/python compare_x_junction.py

# Include logical error rate analysis (requires pymatching)
./venv/bin/python compare_x_junction.py --error-rates

# Custom shots and noise levels
./venv/bin/python compare_x_junction.py --error-rates --shots 100000 --noise-levels 0.001 0.002 0.003
```

Outputs include:
- Graph-like distance for both circuits
- Number of qubits, detectors, and observables
- Crumble URLs for visualization
- Logical error rates and plot (when using `--error-rates`)

#### 3. Flag-Aware Decoding of the Spatial Hadamard (`flag_aware_decoder.py`)

The spatial Hadamard primitive measures flag qubits that herald the hook errors of
the stretched stabilizers. `FlagAwareMatching` makes the flag-free pieces of those
hyperedges conditional on the measured flags, so an unheralded hook is priced as two
faults instead of one. Three strategies are available: `reweight` (one gated matching
call), `iterative` (scored commitment of heralded mechanisms), and `hyperedge`
(the same commitment, but any split mechanism may be proposed).

```bash
# Unit tests
./venv/bin/python -m pytest test_flag_aware_decoder.py

# How many faults each decoder needs before it fails; exhaustive up to --max-weight
./venv/bin/python measure_decoder_fault_distance.py --k 1 --max-weight 2 --include-tesseract
./venv/bin/python measure_decoder_fault_distance.py --k 3 --max-weight 2 --sample 120000

# Logical error rate sweep with an effective-distance fit.  Progress is
# checkpointed, so an interrupted run resumes where it left off.
./venv/bin/python benchmark_flag_aware_decoder.py --k-values 1 2 3 \
    --decoders pymatching correlated_pymatching flag_aware_reweight flag_aware tesseract
./venv/bin/python benchmark_flag_aware_decoder.py --plot-only
```

## Key Features

- **Same Detector Count**: Both diagonal and standard circuits produce the same number of detectors, ensuring each ancilla connects to the same number of data qubits
- **Corner Handling**: Properly handles 3-body corner plaquettes when two adjacent arms are missing in spatial cubes
- **Boundary Plaquettes**: 2-body boundary plaquettes derived from diagonal bulk schedules with appropriate qubit omissions

## Requirements

- Python 3.13+
- Dependencies in `venv/` (activate with `source venv/bin/activate`)
- tqec library (installed in venv)

