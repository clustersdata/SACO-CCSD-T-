# SACO-CCSD-T-
SACO-CCSD(T): A New Efficient CCSD(T) Approach Accelerated by the Symmetry-Adapted and Cache-Optimized Real-Integrals

# README.md
# SACO-CCSD(T) PySCF High-Speed Implementation
## Project Overview
This repository contains two core files for fast symmetry-adapted CCSD(T) single-point energy calculations based on PySCF:
1. `w3_CCSD_T.py`: Main driver script – defines molecular geometry, executes RHF → CCSD → SACO-(T) correction workflow, outputs total CCSD(T) energy and wall-clock runtime.
2. `ccsd_t_saco.py`: Optimized SACO-CCSD(T) correction kernel (custom `ccsd_t_saco()` implementation). Implements symmetry-blocked integral sorting, cache-aware block contraction, C-library low-level tensor contraction, out-of-core HDF5 temporary storage, and OpenMP multi-threading acceleration for real/complex orbital integrals.

Key features of this custom SACO-(T) module:
- Symmetry irrep partitioning of occupied/virtual orbitals to reduce redundant tensor contractions
- CPU cache-aligned block splitting for minimal cache miss overhead
- In-core / out-of-core HDF5 temporary integral storage switching
- C-layer `libcc` tensor contraction (avoids slow pure-Python triple-loop t3 terms)
- Automatic memory buffer size tuning based on available RAM
- OpenMP multi-threading support inherited from PySCF’s underlying C/BLAS/LAPACK stack
- Symmetry recovery & index sorting for Abelian point groups
- Continuous memory array coercion (`ascontiguousarray`) for vectorized CPU execution

---
# Table of Contents
1. Environment & Dependency Installation
2. File Structure & Code Function Breakdown
3. Basic Execution Guide (Run `w3_CCSD_T.py`)
4. Runtime Cache & Memory Parameter Configuration
    4.1 Global PySCF Memory Limit
    4.2 In-Core vs Out-of-Core SACO(T) Storage
    4.3 Cache Line Alignment & Block Buffer Tuning
    4.4 Temporary HDF5 File Control
5. Symmetry Group Detection & SACO Workflow Deep Dive
    5.1 Molecular Symmetry Input Setup
    5.2 Irrep Labeling & Orbital Sort Pipeline (`_sort_eri` / `_sort_t2_vooo_`)
    5.3 Symmetry Blocked Tensor Contraction Logic
    5.4 Symmetry Recovery After Contraction
6. OpenMP Parallelization Setup & Tuning
    6.1 System Environment Variables
    6.2 PySCF Thread Control Interfaces
    6.3 Multi-Threaded Contract Asynchronous Dispatch
    6.4 Thread Scaling Best Practices
7. Output Result Explanation
8. Troubleshooting Common Runtime Errors
9. Benchmark & Test Cases (Built-in `ccsd_t_saco.py` Main Block)

---
# 1. Environment & Dependency Installation
## 1.1 Minimum System Requirements
- OS: Linux (Ubuntu/CentOS recommended); macOS compatible; Windows WSL2 required
- Python ≥3.9
- Compiler: GCC ≥9 / Clang ≥12 (with OpenMP support for BLAS/LAPACK/PySCF libcc)
- RAM: ≥4GB (small sto-3g water trimer test case); ≥16GB for double-zeta basis sets

## 1.2 Core Dependencies
| Package | Purpose | Install Command |
|--------|---------|----------------|
| `pyscf` | Quantum chemistry backend (HF/CCSD integral, libcc C library) | `pip install pyscf` |
| `numpy` | Array storage, vectorized operations, memory contiguous alignment | `pip install numpy>=1.23` |
| `h5py` | Out-of-core temporary HDF5 integral storage (SACO(T) out-of-core mode) | `pip install h5py` |
| `ctypes` | Built-in, C library interface for `libcc` tensor contraction | No install required |
| `scipy` | Optional, for advanced linear algebra symmetry operations | `pip install scipy` |

### 1.3 OpenMP-Enabled PySCF Build (Critical for Parallel Speedup)
Default PySCF pip wheels may disable OpenMP. Recompile PySCF with OpenMP:
```bash
# 1. Clone PySCF source
git clone https://github.com/pyscf/pyscf.git
cd pyscf
# 2. Enable OpenMP build flags
export OMP_CXX_FLAGS="-fopenmp"
export OMP_CC_FLAGS="-fopenmp"
python setup.py build_ext --inplace
pip install .
```
Verify OpenMP compilation:
```python
import pyscf.lib
print(pyscf.lib.num_threads()) # Should return default thread count >1
```

## 1.4 Project File Deployment
Place both files in the same working directory:
```
./
├── w3_CCSD_T.py       # Main calculation driver
└── ccsd_t_saco.py     # Custom SACO-CCSD(T) kernel
```
No additional path modification required – the main script calls `mycc.ccsd_t_saco()` which internally loads the custom kernel when imported into the PySCF CC module context.

---
# 2. File Structure & Code Function Breakdown
## 2.1 `w3_CCSD_T.py` Main Driver
1. Timing wrapper for total CCSD(T) wall-clock measurement
2. Trimer water molecular geometry definition (3 H₂O molecules, sto-3g basis)
3. PySCF `gto.M()` molecule builder: symmetry flag, charge/spin, verbose logging
4. RHF SCF mean-field calculation (`scf.HF(mol).run()`)
5. CCSD amplitude iteration (`cc.CCSD(mf).run()`)
6. Custom SACO-(T) correction energy evaluation via `mycc.ccsd_t_saco()`
7. Sum CCSD correlation energy + (T) perturbative triple correction for total CCSD(T) energy
8. Formatted print output of SCF, CCSD, (T) correction, total energy, runtime

## 2.2 `ccsd_t_saco.py` SACO-CCSD(T) Kernel
### Core Top-Level Function
`kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE)`: Main entry point for SACO(T) triple correction energy calculation
### Supporting Subroutines
1. `_sort_eri()`: Symmetry-sort two-electron integrals into irrep-blocked `eris_vvop` storage
2. `_sort_t2_vooo_()`: Symmetry permute t1/t2 amplitudes, Fock matrix, vooo integrals; provide in-place t2 restoration function
3. `_irrep_argsort()`: Vectorized orbital index sorting by symmetry irreducible representation (irrep)
4. Internal `contract()` closure: C-layer `libcc` tensor contraction driver with precomputed C-type pointers for low overhead
### Key Optimized Logic
- HDF5 temporary dataset for out-of-core integral storage (`lib.H5TmpFile`)
- Cache-aware block buffer size auto-calculation aligned to 64-byte CPU cache lines
- Asynchronous background I/O for integral transposition/saving (`lib.call_in_background`)
- Thread-safe parallel block contraction dispatch
- Imaginary part warning for complex orbital integrals (non-zero imaginary triple correction flag)

---
# 3. Basic Execution Guide
## 3.1 Single Command Run
```bash
# Set OpenMP threads first (see Section 6)
export OMP_NUM_THREADS=8
python w3_CCSD_T.py
```
## 3.2 Modify Input Molecular System
Edit the `atom` multi-line string inside `gto.M()` in `w3_CCSD_T.py`:
- Coordinates in Angstrom units
- Adjust `basis='sto-3g'` to `cc-pVDZ`, `cc-pVTZ`, etc. for larger basis sets
- Toggle `symmetry=True/False` to enable/disable symmetry-adapted SACO optimization
- Modify `charge=N`, `spin=M` for charged/open-shell systems
- `verbose=4`: Increase log verbosity (0 = silent, 5 = full integral print)

## 3.3 Run Built-in Test Cases in `ccsd_t_saco.py`
Two reference water test cases (C₂v symmetry / no symmetry) for validation:
```bash
python ccsd_t_saco.py
```
Output prints the difference between computed SACO(T) correction and reference benchmark values to verify kernel correctness.

---
# 4. Runtime Cache & Memory Parameter Configuration
All memory/cache tuning parameters live in `ccsd_t_saco.py`’s `kernel()` function and PySCF global configuration.

## 4.1 Global PySCF Maximum Memory Limit
PySCF uses `mycc.max_memory` (unit: MB) to allocate space for integrals, t1/t2 tensors, and cache buffers.
### Modify in `w3_CCSD_T.py`
```python
mycc = cc.CCSD(mf)
mycc.max_memory = 16000  # Set 16GB total memory limit for CC module
mycc.run()
```
### Memory Calculation Logic Inside SACO(T) Kernel
```python
mem_now = lib.current_memory()[0]  # MB RAM already consumed by PySCF
max_memory = max(0, mycc.max_memory - mem_now)  # Remainder available for SACO(T)
bufsize = (max_memory * 0.7e6 / elem_size - nocc**3 * 3 * lib.num_threads()) / (nocc * nmo)
```
- `0.7`: Safety coefficient to reserve RAM for OS/Python overhead
- `elem_size`: Bytes per float (8 for float64, 16 for complex128)
- `nocc**3 * 3 * threads`: Reserve buffer for multi-threaded triple tensor temporary storage

## 4.2 In-Core vs Out-of-Core Temporary Integral Storage
Controlled by `mycc.incore_complete` boolean flag:
### In-Core Mode (Faster, RAM-Heavy)
```python
mycc.incore_complete = True
# Kernel behavior: eris_vvop stored as numpy array in system RAM
```
Use for small systems (sto-3g, minimal occupied/virtual orbitals).
### Out-of-Core Mode (Low RAM, HDF5 I/O Overhead)
```python
mycc.incore_complete = False
# Kernel behavior: eris_vvop written to temporary HDF5 via lib.H5TmpFile()
```
Temporary HDF5 files auto-deleted after calculation finishes. Adjust async I/O with `mycc.async_io`:
```python
mycc.async_io = True  # Enable asynchronous disk write for integrals (reduces I/O stall)
```

## 4.3 CPU Cache Line Alignment & Block Buffer Tuning
### Fixed Cache Line Constant (64 Bytes – Standard x86 CPU L1/L2 Cache Line)
```python
cache_line_size = 64
bufsize = (bufsize // cache_line_size) * cache_line_size  # Force buffer alignment
bufsize = max(32, bufsize)  # Minimum outer virtual orbital block size
b_bufsize = max(8, bufsize // 8)  # Smaller inner block for tril nested loops
```
### Manual Override Buffer Size
If auto-calculated `bufsize` causes cache thrashing, hardcode in `ccsd_t_saco.py`:
```python
# Replace auto bufsize calculation with fixed value
bufsize = 256
bufsize = (bufsize // cache_line_size) * cache_line_size
```
Larger `bufsize`: fewer kernel dispatch calls, higher RAM usage; smaller `bufsize`: lower memory footprint, more thread scheduling overhead.

## 4.4 Temporary File Cache Control
- HDF5 temp file location: PySCF default temporary directory (`/tmp` on Linux)
- Change temp directory via environment variable:
```bash
export PYSCF_TMPDIR=/scratch/user/tmp
python w3_CCSD_T.py
```
- Cleanup: All `lib.H5TmpFile` objects are automatically unlinked when the kernel exits; no residual files remain.

---
# 5. Symmetry Group Detection & SACO Workflow Deep Dive
SACO = Symmetry-Adapted Cluster Operator, reduces contraction cost by only computing non-zero symmetry-allowed tensor blocks.

## 5.1 Enable Molecular Symmetry Detection
In `w3_CCSD_T.py` molecule definition:
```python
mol = gto.M(
    atom='''...''',
    basis='sto-3g',
    symmetry = True,  # Critical flag for symmetry workflow
    charge=0,
    spin=0,
    verbose=4
)
```
When `symmetry=False`, all symmetry sorting logic is skipped; kernel runs full unblocked tensor contraction (slower for large systems).
PySCF auto-detects Abelian point groups (C₁, C₂, Cs, C₂v, D₂, D₂h) from input Cartesian coordinates.

## 5.2 Step 1: Irrep Orbital Labeling (`_sort_eri`)
1. Compute overlap matrix `ovlp = mycc._scf.get_ovlp()`
2. `symm.label_orb_symm()` assigns integer irrep ID to each molecular orbital (0–7 for max 8 irreps in D₂h)
3. Split orbitals into occupied (`orbsym[:nocc]`) and virtual (`orbsym[nocc:]`) symmetry labels
4. `_irrep_argsort()` generates index permutations to group orbitals by identical irrep
5. Reorder two-electron integrals `eris_vvop` into contiguous symmetry blocks for cache efficiency

## 5.3 Step 2: Symmetry Permute Amplitudes & Integrals (`_sort_t2_vooo_`)
1. Permute `t1`, `t2`, `vooo` integrals, Fock matrix `fvo` using irrep-sorted orbital indices
2. Reshape t2 amplitude tensor `(nocc,nocc,nvir,nvir)` → symmetry-blocked `(nvir,nvir,nocc,nocc)` `t2T` layout optimized for contraction
3. Precompute cross-irrep XOR lookup table `oo_sym = o_sym[:, None] ^ o_sym`: only orbitals with matching irrep product contribute non-zero triple corrections
4. Precompute cumulative irrep offset arrays `o_ir_loc`, `v_ir_loc`, `oo_ir_loc` passed to C `libcc` contraction function to skip zero symmetry blocks

## 5.4 Step 3: Symmetry-Blocked Tensor Contraction
1. Outer loop over upper-triangular virtual orbital blocks `prange_tril` to avoid redundant symmetric pairs
2. Inner nested triangular virtual block loop for b0/b1 ranges
3. Pass precomputed irrep offset pointers to C `CCsd_t_contract` / `CCsd_t_zcontract`
4. C-layer skips all tensor sub-blocks where irrep direct product symmetry forbids non-zero contribution

## 5.5 Step 4: Symmetry Restoration
`restore_t2_inplace()` closure returns t2 amplitude tensor to original unsorted orbital order after triple correction energy calculation, preserving PySCF standard CCSD object interface compatibility.

## 5.6 Symmetry Debug Logging
Set `verbose=5` in `mol = gto.M(..., verbose=5)` to print full irrep orbital assignments, symmetry block sizes, and skipped zero-contraction blocks during kernel execution.

---
# 6. OpenMP Parallelization Setup & Tuning
Parallelism is implemented at two levels:
1. Low-level BLAS/LAPACK OpenMP for matrix multiplication (PySCF built-in)
2. Multi-threaded asynchronous block contraction dispatch in SACO(T) kernel (`lib.call_in_background`)

## 6.1 Global OpenMP Thread Environment Variables (Set Before Execution)
```bash
# Core thread count control (most critical)
export OMP_NUM_THREADS=8
# Thread scheduling policy
export OMP_SCHEDULE=static
# Bind threads to CPU cores (Linux only, for multi-socket nodes)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Disable nested OpenMP (prevents oversubscription)
export OMP_NESTED=FALSE
```
### Thread Oversubscription Warning
If `OMP_NUM_THREADS` exceeds physical CPU cores, severe slowdown occurs; match thread count to physical CPU cores.

## 6.2 PySCF Internal Thread Control API
Override environment variables within Python script:
```python
import pyscf.lib
pyscf.lib.set_num_threads(8)
print(pyscf.lib.num_threads()) # Confirm thread count
```
## 6.3 Asynchronous Multi-Threaded Contraction Dispatch
```python
with lib.call_in_background(contract, sync=not mycc.async_io) as async_contract:
    # Outer virtual block loop
    for a0,a1 in prange_tril_range:
        async_contract(...)
        # Inner b block loop
        for b0,b1 in lib.prange_tril(...):
            async_contract(...)
```
- `sync=False` (`mycc.async_io=True`): Background thread pool dispatches multiple contraction blocks simultaneously, overlaps computation and disk I/O (out-of-core mode)
- `sync=True`: Serial block execution (single thread, debug only)

## 6.4 Parallel Scaling Best Practices
1. Small systems (sto-3g water trimer): Threads 2–4 minimal overhead; >6 threads no speed gain due to small block size
2. Medium/large basis (cc-pVDZ/cc-pVTZ): Threads equal to physical CPU cores (8–16) show linear scaling
3. Out-of-core mode: Limit threads to ≤ half physical cores to avoid I/O bandwidth contention
4. Disable hyper-threading for heavy CC calculations (floating-point bound workloads benefit little from SMT)

---
# 7. Output Result Explanation
print block from `w3_CCSD_T.py`:
```
## Log Output Control
`mol.verbose=4`: Prints SCF convergence, CCSD amplitude iteration convergence, SACO(T) symmetry block timing breakdowns.
`mol.verbose=2`: Minimal output, only final energy values.

---
# 8. Troubleshooting Common Runtime Errors
## 8.1 High Memory Usage / Out-of-Memory Kill
1. Reduce `mycc.max_memory` value
2. Switch `mycc.incore_complete = False` to enable out-of-core HDF5 storage
3. Lower `OMP_NUM_THREADS` to reduce per-thread temporary buffer allocation
4. Use smaller basis set (sto-3g → cc-pVDZ)

## 8.2 Slow Runtime with Symmetry Enabled
1. Verify molecule coordinates are geometrically symmetric (small coordinate noise breaks irrep detection)
2. Increase buffer `bufsize` to reduce kernel dispatch overhead
3. Set `mycc.async_io=True` to overlap I/O and computation

## 8.3 Non-Zero Imaginary Part Warning
```
WARN: Non-zero imaginary part of CCSD(T) energy was found X.XX
```
- Cause: Complex orbital integrals, broken symmetry orbitals, or numerical linear dependency
- Fix: Re-run SCF with tighter convergence `mf.conv_tol = 1e-12`; verify real geometry with real basis functions

## 8.4 OpenMP No Speedup
1. Recompile PySCF with OpenMP enabled (Section 1.3)
2. Confirm `OMP_NUM_THREADS` set before Python launch
3. Disable nested parallelism with `export OMP_NESTED=FALSE`

## 8.5 HDF5 Temporary File Permission Denied
1. Modify `PYSCF_TMPDIR` to a user-writable scratch directory
2. Increase disk space on temporary storage partition

---
# 9. Benchmark Test Cases Built Into `ccsd_t_saco.py`
Two validation benchmarks execute when running `python ccsd_t_saco.py`:
1. C₁ symmetry H₂O (no symmetry optimization): Reference E(T) = -0.0033300722704016289
2. C₂v symmetric H₂O (full SACO symmetry block contraction): Reference E(T) = -0.003060022611584471

The script prints the numerical difference between computed correction and benchmark value to validate kernel implementation correctness. Differences <1e-10 confirm proper integral sorting, symmetry handling, and C-layer contraction logic.
