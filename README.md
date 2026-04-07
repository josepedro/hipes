# HiPES: High-Throughput Chunk-Based GPU Bitmap Compression

This repository contains the explicit source code, baseline validations, and synthetically modeled datasets developed to evaluate **HiPES (Hierarchical Performant and Efficient Set)**, as formalized in our ICCS 2026 proceedings.

HiPES is a GPU-native, chunk-oriented compressed bitmap scheme intentionally designed to bypass sequential variable-length encoding bottlenecks found in legacy CPU schemes. By strictly aligning with SIMT execution hardware limits and shared-memory (SMEM) constraints, HiPES achieves zero-divergence coalesced access, enabling high throughput set intersections necessary for Graph Neural Network (GNN) pre-processing and massive-scale relational filtering.

## Repository Contents

A streamlined distribution of the 5 distinct computational artifacts evaluated in the study. Internal script comments have been fully stripped for structural clarity.

- **`hipes_core.cu`**
  The robust C++ CUDA logic encompassing the SIMT pass-based allocations and algorithmic atomic groupings defining the chunk boundaries without relying on global metadata contention.
  
- **`gpu_wah_baseline.cu`**
  The baseline benchmark engine (GPU-WAH). Adapted to leverage parallel Thrust implementations for accurate, symmetric evaluations of cross-word boundary thresholds against HiPES.

- **`dataset_generator.cpp`**
  The reproducible, multithreaded simulation framework used to produce strictly enforced combinatorial test bounds (modulating Uniform vs. Clustered localities targeting $N=10^6$/$10^7$ variables and $E=10^8$/$10^9$ cardinality spaces). Modalities bypass exponential hash collisions using $O(1)$ dynamic vector bitsets.

- **`data/`**
  The complete resulting archive of the mathematical generator script storing identical copies of the 8 uniform bounded scenarios evaluated throughout our tests.

- **`artifact_architecture_cross_validation.pdf`**
  The formal supplemental artifact mapping cross-architecture methodological scaling evaluations, pseudo-code bounds, and legacy evaluation tables referenced inside our primary manuscript.
