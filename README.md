# Random Forests Acceleration with MPI
Goal: To accelerate a sequential implementation of the Random Forests method using MPI

Random Forests build an "ensemble" (a group) of many individual decision trees to make more accurate predictions for both classification (e.g., spam/not spam) and regression (e.g., predicting house prices) tasks, by using random subsets of data and features to train each tree, then 
combining their results through voting or averaging for a robust final answer. They reduce the variance and overfitting issues of single decision trees, making them stable and reliable algorithms. How They Work: 

1) Bootstrap Aggregation (Bagging): Each tree in the forest is trained on a random 
subset of the original training data (sampled with replacement);
2) Feature Randomness: At each split point in a tree, only a random subset of features (variables) is considered, not all of them, which decorrelates the trees;
3) Ensemble Voting/Averaging: in Classification, the final prediction is determined by majority vote (the class predicted by most trees); in Regression, the final prediction is the average of the predictions from all the individual trees.

**Advisor:** <a href="https://www.cienciavitae.pt/portal/C414-F47F-6323">José Carlos Rufino Amaro</a><br/>
**Educational Institution:** [Instituto Politécnico de Bragança](IPB.pt)

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Compilation](#installation--compilation)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Profiling](#profiling)
- [Benchmarking](#benchmarking)
- [Results](#results)
- [Discussion](#discussion)
- [References](#references)

---

## 🎯 Overview

This project accelerates a sequential implementation of the **Random Forests** machine learning algorithm using **MPI (Message Passing Interface)** for parallel computing. Random Forests build an ensemble of decision trees to make accurate predictions by training each tree on random data subsets and combining their results through voting.

### Project Evolution

1. **Original Serial Version** - Baseline implementation with memory bottlenecks
2. **Optimized Serial Version** - Removed memory allocation hotspots (3x faster)
3. **MPI Parallel Version** - Tree-level parallelization achieving near-linear speedup

### Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset in CSV format:
- **Features**: Medical image measurements for breast cancer diagnosis
- **Classes**: Binary classification (0 = Malignant, 1 = Benign)
- **Samples**: 568 rows × 32 columns

---

## 🔧 Requirements

### Software Dependencies

- **GCC**: `gcc` version 4.8+ (C compiler)
- **OpenMPI**: `mpicc` version 1.10+ (MPI compiler wrapper)
- **gprofng**: GNU profiler (for performance analysis)
- **Make**: Build automation tool

### Hardware Requirements

- **Minimum**: 2 CPU cores
- **Recommended**: 8+ CPU cores for optimal parallelization
- **Memory**: ~500MB RAM

### Installing Dependencies

#### Debian/Ubuntu
```bash
# Check if MPI is installed
mpicc --version
mpirun --version

# Install if needed
sudo apt-get update
sudo apt-get install build-essential openmpi-bin libopenmpi-dev
```

#### Red Hat/CentOS/Rocky Linux
```bash
sudo yum install gcc openmpi openmpi-devel

# Load MPI module
module load mpi/openmpi-x86_64
```

#### Verify Installation

```bash
# Test MPI
mpirun --version
# Expected: mpirun (Open MPI) 4.x.x

# Test compiler
mpicc --version
# Expected: gcc (GCC) 8.x.x or higher
```

---

## 🛠️ Installation & Compilation

#### Step 1: Clone/Extract the Project

```bash
# From ZIP
unzip random-forests-mpi.zip
cd randforest-par-mpi

# Or from Git
git clone <repository-url>
cd randforest-par-mpi
```
#### Step 2: Compile the MPI Version

```bash
# Clean previous builds
make clean

# Compile with optimizations
make

# Verify executable was created
ls -lh random-forest
```
#### Expected output:

```text
-rwxr-xr-x 1 user group 156K Jan 2 15:30 random-forest

```

#### Step 3: Create MPI Hostfile (Cluster Only)

If using a cluster, create `cluster.OPENMPI`:

```bash
# Example for 2 nodes with 8 cores each
echo "rocks-node-7 slots=8 max-slots=8" > cluster.OPENMPI
echo "rocks-node-0 slots=8 max-slots=8" >> cluster.OPENMPI

# Verify
cat cluster.OPENMPI
```

## 🚀 Usage

### Basic Execution

#### Serial Version (1 process)

```bash
./random-forest wdbc.csv --seed 0
```

#### Parallel Version (Local Machine)

```bash
# 2 processes
mpirun -np 2 ./random-forest wdbc.csv --seed 0

# 4 processes
mpirun -np 4 ./random-forest wdbc.csv --seed 0
```

#### Parallel Version (Cluster with Hostfile)

```bash
# 4 processes across nodes
mpirun -np 4 -hostfile cluster.OPENMPI ./random-forest wdbc.csv --seed 0

# 8 processes across 2 nodes
mpirun -np 8 -hostfile cluster.OPENMPI ./random-forest wdbc.csv --seed 0
```

### Command Line Options

```bash
./random-forest <dataset.csv> [OPTIONS]

Options:
  --seed N          Set random seed (default: time-based random)
  --num_rows N      Specify dataset rows (optional, auto-detect)
  --num_cols N      Specify dataset columns (optional, auto-detect)
  --log_level N     Verbosity level:
                    0 = minimal output
                    1 = normal (default)
                    2 = verbose/debug
```

### Usage Examples

```bash
# Fixed seed for reproducibility
mpirun -np 4 ./random-forest wdbc.csv --seed 0

# Verbose logging for debugging
mpirun -np 2 ./random-forest wdbc.csv --seed 0 --log_level 2

# Serial with custom seed
./random-forest wdbc.csv --seed 12345
```


### Expected Output

```
using:
  seed: 0
  verbose log level: 1
  rows: 568, cols: 32
reading from csv file:
  "wdbc.csv"
using:
  k_folds: 10
using RandomForestParameters:
  n_estimators: 20
  max_depth: 7
  min_samples_leaf: 3
  max_features: 10
Distributing 10 trees among 2 processes
cross validation accuracy: 95.789474% (95%)
(time taken: 98.220000s)
```

---

## ⚙️ Configuring Hyperparameters

### Modify Before Compilation

Edit `main.c` (around line 50-60):

```c
const int k_folds = 10;              // Cross-validation folds

const RandomForestParameters params = {
    .n_estimators = 20,              // Number of trees in forest
    .max_depth = 7,                  // Maximum tree depth
    .min_samples_leaf = 3,           // Min samples per leaf node
    .max_features = 10               // Features sampled per tree
};
```

### Recompile After Changes

⚠️ **Important**: Always recompile after changing hyperparameters:

```bash
make clean
make
```

### Parameter Guidelines

| Parameter | Recommended | Impact | Notes |
| :-- | :-- | :-- | :-- |
| `k_folds` | 5, 10 | CV accuracy vs speed | Higher = slower but more reliable |
| `n_estimators` | 10, 20, 30 | Model accuracy | More trees = better but slower |
| `max_depth` | 5, 7, 10 | Tree complexity | Too high = overfitting risk |
| `max_features` | 10, 15, 20 | Feature sampling | Balance speed/accuracy |

**For Benchmarking**: Use `n_estimators = 20` to match report results .

---

## 🔍 Profiling

### 1. Profile Original Serial Version

```bash
cd randforest-serial

# Edit main.c to set:
# k_folds = 10, n_estimators = 10, max_features = 10

make clean && make

# Collect profiling data (~65 seconds)
gprofng collect app -o random-forest.er ./random-forest wdbc.csv --seed 0

# Display function execution times (sorted by inclusive CPU time)
gprofng display text -functions random-forest.er

# Display call tree (functions using >30% CPU time)
gprofng display text -calltree random-forest.er | grep -E "^( {0,10}[0-9]|Name)"
```


### 2. Identified Bottleneck

**Original Version Problem** :

```
calculate_best_data_split (99.99% CPU time)
  └─> split_dataset (99.03%)
       └─> realloc (60.89%) ❌ HOTSPOT
            └─> memcpy (10.22%)
```

**Issue**: `realloc()` called inside loop → massive memory allocation overhead

### 3. Profile Optimized Serial Version

```bash
cd randforest-serial-opt

# After fixing realloc issue
make clean && make
gprofng collect app -o random-forest-opt.er ./random-forest wdbc.csv --seed 0

# View optimized call tree
gprofng display text -calltree random-forest-opt.er | grep -E "^( {0,10}[0-9]|Name)"
```

**Result**: Execution time **66.06s → 21.47s** (3x speedup!) 

### 4. Optimization Applied

**Before** (Bad):

```c
// Allocate small, then realloc in loop
double **left = malloc(1 * sizeof(double*));
for (size_t i = 0; i < rows; i++) {
    if (condition) {
        left = realloc(left, ++left_count * sizeof(double*)); // ❌ Slow!
    }
}
```

**After** (Good):

```c
// Pre-count sizes first
size_t left_count = 0, right_count = 0;
for (size_t i = 0; i < rows; i++) {
    if (condition) left_count++;
    else right_count++;
}

// Allocate once with correct size
double **left = malloc(left_count * sizeof(double*));  // ✅ Fast!
double **right = malloc(right_count * sizeof(double*));
```


---

## 📊 Benchmarking

### Methodology

1. Set `n_estimators = 20` in `main.c`
2. Compile: `make clean && make`
3. Run **3 times** for each N processes
4. Wait **5 seconds** between runs
5. Calculate **mean** and **RSD (Relative Standard Deviation)**
6. Discard runs if RSD > 10% [file:11]

## 📈 Results

### Table 1: Theoretical Predictions (Amdahl's Law)

**Assumptions**: p = 99.99% (parallelizable fraction)


| N | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **S_T** | 1.99 | 3.99 | 5.99 | 7.99 | 9.99 | 11.98 | 13.98 | 15.97 |
| **E_T (%)** | 99.99 | 99.97 | 99.96 | 99.93 | 99.91 | 99.89 | 99.87 | 99.84 |

**Theoretical Speedup Limit**: S_T(∞) = 1/(1-0.9999) = **10,000** 

---

### Table 2: Real Execution Times

| N | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|----|----|----|----|
| **T_R (s)** | 192.35 | 98.22 | 50.27 | 40.57 | 30.23 | 20.69 | 20.64 | 20.57 | 20.48 |
| **RSD (%)** | 0.10 | 0.17 | 0.34 | 1.24 | 0.35 | 0.64 | 0.31 | 0.48 | 0.43 |

✅ All RSD < 1.5% → **Excellent consistency** 

---

### Table 3: Real Performance Metrics

| N | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **S_R** | 1.96x | 3.83x | 4.74x | 6.36x | 9.30x | 9.32x | 9.35x | 9.39x |
| **E_R (%)** | 97.92 | 95.65 | 79.02 | 79.53 | 93.00 | 77.65 | 66.78 | 58.70 |
| **E_R/E_T (%)** | 97.93 | 95.68 | 79.05 | 79.59 | 93.07 | 77.73 | 66.87 | 58.79 |

**Legend**:

- **S_R**: Real speedup
- **E_R**: Real efficiency
- **E_R/E_T**: Closeness to theoretical efficiency

---

### Visualizations

#### Graph 1: Execution Time

```
200s ┤●
     │
150s ┤
     │
100s ┤ ●
     │
 50s ┤   ●  ●
     │         ●───●───●───●
  0s └─────────────────────────
     1  2  4  6  8 10 12 14 16
           Processes (N)
```


#### Graph 2: Speedup (Real vs Ideal)

```
16x ┤            ╱ Ideal
    │          ╱
12x ┤        ╱   ●●●● Real
    │      ╱   ●╱
 8x ┤    ╱   ●╱
    │  ╱   ●╱
 4x ┤╱   ●╱
    │  ●╱
 0x └────────────────────
    2  4  6  8  10 12 14 16
```


#### Graph 3: Efficiency

```
100% ┤●●
     │   ●      ●
 80% ┤     ●  ●   ●
     │              ●
 60% ┤                ●●
     │
 40% ┤
     └────────────────────
     2  4  6  8  10 12 14 16
```


---

## 💡 Discussion

### Key Findings 

✅ **Excellent for Divisible Workloads**

- **N = 2, 4, 10**: Efficiency ≥ 93%
- Near-perfect speedup (close to theoretical predictions)
- 20 trees ÷ 2/4/10 = balanced distribution

❌ **Efficiency Drops for Non-Divisible Workloads**

- **N = 6, 8, 12, 14, 16**: Efficiency 58-79%
- Load imbalance causes idle processes


### Example: 20 Trees with 6 Processes

```
Process 0: [4 trees] ████████████ 
Process 1: [4 trees] ████████████
Process 2: [3 trees] █████████ ← Finishes early
Process 3: [3 trees] █████████ ← Idle waiting
Process 4: [3 trees] █████████ ← Idle waiting
Process 5: [3 trees] █████████ ← Idle waiting
                              ↓
                    Waiting at MPI_Allreduce
```

4 processes finish early and wait → **wasted CPU time**

### Why Divergence from Amdahl's Law?

1. **Static Load Imbalance**: When `n_estimators % N ≠ 0`
2. **Synchronization Overhead**: `MPI_Bcast` and `MPI_Allreduce` cost
3. **Idle Time**: Not predicted by Amdahl (assumes perfect distribution)
4. **Communication Latency**: More processes = more coordination 

### Bonus Test: Perfect Balance

| Configuration | Result |
| :-- | :-- |
| 20 trees, 20 processes | T = 10.82s |
| Speedup | 17.77x |
| Efficiency | **88.88%** ✅ |
| E_R/E_T | 89.05% |

**Conclusion**: Efficiency jumps from 58.7% (N=16) to 88.9% (N=20) when perfectly balanced! 