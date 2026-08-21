# ML Framework in C From Scratch

A minimal machine-learning framework implemented entirely in C to explore how neural-network libraries work internally.

The project builds the fundamental components of a neural-network framework without relying on libraries such as PyTorch, TensorFlow, NumPy, or Eigen for the actual model implementation.

The framework includes:

* A custom matrix implementation
* Matrix arithmetic and linear algebra operations
* ReLU and Softmax activations
* Cross-entropy loss
* Computational graphs
* Automatic differentiation / backpropagation
* Model variables and operation tracking
* Gradient accumulation
* Parameter updates
* Custom memory arenas
* Pseudo-random number generation
* MNIST data loading and training

The project is primarily intended as a **learning and systems-programming exercise** for understanding the mechanics behind machine-learning frameworks.

---

## Table of Contents

* [Motivation](#motivation)
* [Project Architecture](#project-architecture)
* [Repository Structure](#repository-structure)
* [Core Components](#core-components)

  * [Base Types](#1-base-types)
  * [Memory Arena](#2-memory-arena)
  * [Random Number Generator](#3-random-number-generator)
  * [Matrix Engine](#4-matrix-engine)
  * [Computational Graph](#5-computational-graph)
  * [Automatic Differentiation](#6-automatic-differentiation)
  * [Model Training](#7-model-training)
  * [MNIST Pipeline](#8-mnist-pipeline)
* [Matrix Operations](#matrix-operations)
* [Neural Network Operations](#neural-network-operations)
* [Automatic Differentiation](#automatic-differentiation)
* [Memory Management](#memory-management)
* [MNIST Dataset](#mnist-dataset)
* [Building the Project](#building-the-project)
* [Running the Project](#running-the-project)
* [Implementation Walkthrough](#implementation-walkthrough)
* [Learning Objectives](#learning-objectives)
* [Current Limitations](#current-limitations)
* [Future Improvements](#future-improvements)
* [License](#license)
* [Author](#author)

---

# Motivation

Machine-learning frameworks provide high-level APIs that hide most of the underlying computation.

For example, a framework such as PyTorch allows a user to write:

```python
loss.backward()
```

and the framework automatically constructs and traverses a computational graph, calculates gradients, and propagates them through the model.

This project asks:

> **What would it take to implement those mechanisms ourselves in C?**

Instead of using an existing tensor library or automatic-differentiation engine, this project implements the underlying components directly.

The goal is not to compete with production frameworks.

The goal is to understand them.

---

# Project Architecture

The framework can be viewed as several layers.

```text
                         ┌──────────────────────┐
                         │      MNIST Data      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      mnist.py        │
                         │ Data preparation      │
                         └──────────┬───────────┘
                                    │
                              Binary data
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────┐
│                    C ML Framework                        │
│                                                         │
│  ┌──────────────┐      ┌─────────────────────────────┐  │
│  │   Matrix     │      │     Computational Graph     │  │
│  │   Engine     │◄────►│                             │  │
│  └──────┬───────┘      │ Inputs → Operations → Loss │  │
│         │              └─────────────┬───────────────┘  │
│         │                            │                  │
│         ▼                            ▼                  │
│  Matrix Operations            Backpropagation           │
│  ReLU                        Gradient Computation       │
│  Softmax                     Parameter Gradients        │
│  MatMul                                                │
│  Cross Entropy                                          │
│                                                         │
│              ┌─────────────────────────┐                │
│              │     Memory Arena        │                │
│              └─────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         Updated Model Parameters
```

---

# Repository Structure

```text
ML-Framwork-in-C-from-Scratch/
│
├── main.c
│
├── base.h
│
├── arena.c
├── arena.h
│
├── prng.c
├── prng.h
│
└── mnist.py
```

## `main.c`

The main implementation of the framework.

It contains:

* Matrix representation
* Matrix operations
* Neural-network operations
* Model variables
* Computational graph representation
* Forward execution
* Gradient computation
* Backpropagation
* Training logic

The matrix is represented as a row-major 2D array:

```c
typedef struct {
    u32 rows, cols;
    f32* data;
} matrix;
```

---

## `base.h`

Contains common types and utility definitions used throughout the project.

Examples include:

```c
typedef int32_t i32;
typedef int64_t i64;

typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
```

It also provides utility macros such as:

```c
KiB(n)
MiB(n)
GiB(n)

MIN(a, b)
MAX(a, b)
ALIGN_UP_POW2(n, p)
```

This gives the rest of the framework a consistent low-level type system.

---

## `arena.c` / `arena.h`

Implements the custom memory arena.

Instead of repeatedly allocating and freeing individual objects, the framework can reserve a large region of memory and allocate objects from it.

The API includes:

```c
arena_create()
arena_destroy()

arena_push()
arena_pop()
arena_pop_to()

arena_clear()

arena_temp_begin()
arena_temp_end()

arena_scratch_get()
arena_scratch_release()
```

Convenience macros are also provided:

```c
PUSH_STRUCT()
PUSH_STRUCT_NZ()

PUSH_ARRAY()
PUSH_ARRAY_NZ()
```

---

## `prng.c` / `prng.h`

Provides pseudo-random number generation based on the **PCG family of random-number generators**.

The interface supports both global and explicit RNG state:

```c
prng_seed()
prng_seed_r()

prng_rand()
prng_rand_r()

prng_randf()
prng_randf_r()
```

Random values are used by the framework for tasks such as parameter initialization.

---

## `mnist.py`

Prepares the MNIST dataset for the C implementation.

It uses TensorFlow Datasets to retrieve MNIST and converts the resulting dataset into NumPy arrays.

The images are normalized:

```python
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0
```

The arrays are then written directly to binary files using `tofile()`.

---

# Core Components

## 1. Base Types

The project defines short aliases for fixed-width integer and floating-point types.

```text
i8      signed 8-bit integer
i16     signed 16-bit integer
i32     signed 32-bit integer
i64     signed 64-bit integer

u8      unsigned 8-bit integer
u16     unsigned 16-bit integer
u32     unsigned 32-bit integer
u64     unsigned 64-bit integer

f32     32-bit floating point
```

This makes the low-level implementation easier to read and provides explicit control over data sizes.

---

# 2. Memory Arena

Memory management is an important part of the project.

A traditional approach would allocate every matrix and model variable separately:

```c
malloc(...)
free(...)
```

Instead, this project uses an arena allocator.

The arena maintains:

```c
typedef struct {
    u64 reserve_size;
    u64 commit_size;

    u64 pos;
    u64 commit_pos;
} mem_arena;
```

The `pos` field tracks the current allocation position.

Conceptually:

```text
Arena

+------------------------------------------------------+
| object | object | matrix | graph node | temporary   |
+------------------------------------------------------+
                         ^
                         |
                        pos
```

Allocating memory advances the position.

Releasing memory can simply move the position backwards.

This is particularly useful for computational graphs because many objects have the same lifetime as the model itself.

---

# 3. Random Number Generator

The framework includes a PCG-based random-number generator.

The state is represented by:

```c
typedef struct {
    u64 state;
    u64 inc;
} prng_state;
```

The API provides:

```c
u32 prng_rand(void);
f32 prng_randf(void);
```

and state-specific versions:

```c
u32 prng_rand_r(prng_state* rng);
f32 prng_randf_r(prng_state* rng);
```

This allows the framework to generate reproducible random values when the RNG is seeded appropriately.

---

# 4. Matrix Engine

The matrix engine is the numerical foundation of the framework.

A matrix consists of:

```c
typedef struct {
    u32 rows;
    u32 cols;
    f32* data;
} matrix;
```

Data is stored in **row-major order**.

For a matrix:

```text
[a b c]
[d e f]
```

the underlying memory is:

```text
[a, b, c, d, e, f]
```

This provides a simple contiguous representation.

---

# Matrix Operations

The framework provides operations including:

### Creation

```c
mat_create()
```

Creates a matrix using memory from the arena.

### Loading

```c
mat_load()
```

Loads matrix data from a file.

### Copying

```c
mat_copy()
```

Copies one matrix into another.

### Clearing

```c
mat_clear()
```

Sets matrix values to zero.

### Filling

```c
mat_fill()
```

Fills a matrix with a constant.

### Random initialization

```c
mat_fill_rand()
```

Initializes matrix elements with random values in a specified range.

### Scaling

```c
mat_scale()
```

Multiplies every matrix element by a scalar.

### Summation

```c
mat_sum()
```

Computes the sum of matrix elements.

### Argmax

```c
mat_argmax()
```

Returns the index of the largest element.

---

# Matrix Arithmetic

The framework implements:

```c
mat_add()
mat_sub()
mat_mul()
```

Matrix multiplication supports additional options:

```c
mat_mul(
    matrix* out,
    const matrix* a,
    const matrix* b,
    b8 zero_out,
    b8 transpose_a,
    b8 transpose_b
);
```

The transpose flags allow the same multiplication routine to handle several combinations without explicitly constructing transposed matrices.

---

# Neural Network Operations

The framework currently implements several fundamental neural-network operations.

## ReLU

```c
mat_relu()
```

The ReLU function is:

```text
f(x) = max(0, x)
```

Example:

```text
Input:

[-2  -1   0   2   4]

ReLU:

[ 0   0   0   2   4]
```

---

## Softmax

```c
mat_softmax()
```

Softmax converts a vector of logits into a probability distribution.

For logits `z`:

```text
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

The resulting values satisfy:

```text
0 <= p_i <= 1
```

and:

```text
Σ p_i = 1
```

This is useful for multi-class classification such as MNIST.

---

## Cross-Entropy

```c
mat_cross_entropy()
```

Cross-entropy measures the difference between predicted probabilities and target probabilities.

For a target distribution `q` and prediction `p`:

```text
L = -Σ q_i log(p_i)
```

For classification problems, the target is typically represented as a one-hot vector.

---

# Computational Graph

One of the most important parts of the project is the computational graph.

Instead of treating every operation independently, the framework represents computations using `model_var` objects.

A model variable contains:

```c
typedef struct model_var {
    u32 index;
    u32 flags;

    matrix* val;
    matrix* grad;

    model_var_op op;

    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;
```

Each variable therefore stores:

* Its index
* Flags
* Current value
* Gradient
* Operation that produced it
* Input variables

This creates a directed computation graph.

For example:

```text
Input X
   │
   ▼
Matrix Multiply
   │
   ▼
   W
   │
   ▼
 ReLU
   │
   ▼
Softmax
   │
   ▼
Prediction
   │
   ▼
Cross Entropy
   │
   ▼
  Loss
```

---

# Model Operations

The framework defines operations using an enumeration:

```c
typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,

    MV_OP_RELU,
    MV_OP_SOFTMAX,

    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
} model_var_op;
```

The operation type determines how the variable is calculated during the forward pass and how gradients are propagated during the backward pass.

---

# Model Variable Flags

Variables can have different roles.

The framework defines flags including:

```c
MV_FLAG_REQUIRES_GRAD
MV_FLAG_PARAMETER
MV_FLAG_INPUT
MV_FLAG_OUTPUT
MV_FLAG_DESIRED_OUTPUT
MV_FLAG_COST
```

For example:

```text
Input
  │
  ├── INPUT
  │
  ▼
Weights
  │
  ├── PARAMETER
  ├── REQUIRES_GRAD
  │
  ▼
Prediction
  │
  ├── OUTPUT
  │
  ▼
Loss
  │
  └── COST
```

This allows the training system to distinguish model inputs, parameters, outputs, targets, and the cost function.

---

# Automatic Differentiation

The framework implements the fundamental mechanism behind backpropagation.

Every differentiable model variable can have an associated gradient:

```c
matrix* grad;
```

During the forward pass, the framework calculates values.

During the backward pass, it calculates derivatives with respect to those values.

Conceptually:

```text
Forward:

x ──► operation ──► y ──► loss


Backward:

∂L/∂x ◄── operation gradient ◄── ∂L/∂y ◄── ∂L/∂loss
```

---

# Gradient Operations

The matrix implementation includes explicit gradient functions.

### ReLU gradient

```c
mat_relu_add_grad()
```

### Softmax gradient

```c
mat_softmax_add_grad()
```

### Cross-entropy gradient

```c
mat_cross_entropy_add_grad()
```

The naming convention reflects the fact that these operations accumulate their contribution into an existing gradient.

---

# Forward Pass

The forward pass evaluates the computational graph in dependency order.

For a simple classifier:

```text
Input
  │
  ▼
MATMUL
  │
  ▼
RELU
  │
  ▼
MATMUL
  │
  ▼
SOFTMAX
  │
  ▼
Prediction
  │
  ▼
CROSS_ENTROPY
  │
  ▼
Loss
```

Each operation produces a matrix value stored inside its corresponding model variable.

---

# Backward Pass

After computing the loss, the framework traverses the graph in reverse.

For example:

```text
Loss
 │
 ▼
Cross Entropy
 │
 ▼
Softmax
 │
 ▼
Matrix Multiply
 │
 ▼
ReLU
 │
 ▼
Matrix Multiply
 │
 ▼
Parameters
```

At every operation, the corresponding gradient function computes how the incoming gradient should be distributed to its inputs.

This is the basic mechanism of reverse-mode automatic differentiation.

---

# Model Context

The framework groups the model state into a model context.

The context tracks:

```c
typedef struct {
    u32 num_vars;

    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog;
} model_context;
```

This provides references to the important nodes in the model graph.

It also stores programs representing the forward and cost computations.

---

# Model Training

Training is represented by:

```c
typedef struct {
    matrix* train_images;
    matrix* train_labels;

    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;

    f32 learning_rate;
} model_training_desc;
```

The training configuration therefore contains:

* Training data
* Training labels
* Test data
* Test labels
* Number of epochs
* Batch size
* Learning rate

The general training process is:

```text
Load Dataset
     │
     ▼
Initialize Parameters
     │
     ▼
Create Computational Graph
     │
     ▼
      ┌───────────────┐
      │ Training Loop │
      └───────┬───────┘
              │
              ▼
        Select Batch
              │
              ▼
         Forward Pass
              │
              ▼
        Compute Loss
              │
              ▼
       Backward Pass
              │
              ▼
      Compute Gradients
              │
              ▼
       Update Parameters
              │
              ▼
        Next Batch/Epoch
```

---

# MNIST Dataset

The project uses the MNIST handwritten-digit dataset.

MNIST contains grayscale images of handwritten digits from:

```text
0 → 9
```

Each image is:

```text
28 × 28 pixels
```

The Python preprocessing script uses TensorFlow Datasets:

```python
tfds.load(
    "mnist",
    split=["train", "test"],
    as_supervised=True
)
```

The data is converted into NumPy arrays.

Images are converted to `float32` and normalized:

```python
images.astype(np.float32) / 255.0
```

Labels are also converted to `float32`.

The resulting arrays are written as raw binary data.

---

# Preparing MNIST

Install the required Python packages:

```bash
pip install numpy tensorflow-datasets
```

Then run:

```bash
python mnist.py
```

This generates:

```text
train_images.mat
train_labels.mat
test_images.mat
test_labels.mat
```

Despite the `.mat` extension, these files are **raw binary arrays**, not MATLAB `.mat` files. They are produced using NumPy's `tofile()`.

---

# Building the Project

The framework is written in C and does not require a high-level ML framework for the model itself.

A GCC build can be performed with:

```bash
gcc main.c -o ml_framework -lm
```

Run:

```bash
./ml_framework
```

On Windows using MinGW:

```bash
gcc main.c -o ml_framework.exe -lm
```

Then:

```bash
ml_framework.exe
```

The exact compilation command may need to be adapted to the operating system and compiler because the memory-arena implementation contains platform-specific functionality.

---

# Running the MNIST Example

The typical workflow is:

```bash
# 1. Prepare the dataset
python mnist.py

# 2. Compile
gcc main.c -o ml_framework -lm

# 3. Run
./ml_framework
```

The C program can then load the generated binary dataset and use it as input to the model.

---

# Implementation Walkthrough

If you want to understand the project rather than simply run it, the recommended order is:

## Step 1 — Read `base.h`

Understand:

* Fixed-width types
* Floating-point type
* Boolean aliases
* Utility macros

This establishes the vocabulary used throughout the code.

---

## Step 2 — Read `arena.h`

Understand the memory model:

```text
reserve
  ↓
commit
  ↓
push allocations
  ↓
temporary allocations
  ↓
pop/reset
```

This is important because most framework objects are allocated through the arena.

---

## Step 3 — Read `arena.c`

Study how the arena interacts with the operating system to reserve, commit, decommit, and release memory.

This section is particularly useful for understanding systems programming in C.

---

## Step 4 — Study `matrix`

Start with:

```c
typedef struct {
    u32 rows, cols;
    f32* data;
} matrix;
```

Then follow:

```text
mat_create
    ↓
mat_fill
    ↓
mat_add / mat_sub
    ↓
mat_mul
    ↓
mat_relu
    ↓
mat_softmax
    ↓
mat_cross_entropy
```

At this point you have the numerical foundation of the framework.

---

## Step 5 — Study `model_var`

The next abstraction is the computational graph.

Focus on:

```c
model_var
```

and:

```c
model_var_op
```

Understand how each variable remembers:

```text
value
gradient
operation
inputs
```

---

## Step 6 — Follow the Forward Pass

Trace a computation from input to loss.

For example:

```text
Input
 ↓
MATMUL
 ↓
RELU
 ↓
MATMUL
 ↓
SOFTMAX
 ↓
CROSS_ENTROPY
 ↓
Loss
```

---

## Step 7 — Follow Backpropagation

Start from the loss and move backwards.

At each operation ask:

> Given the gradient of the output, how do we calculate the gradient of the inputs?

This leads directly to:

```c
mat_relu_add_grad()
mat_softmax_add_grad()
mat_cross_entropy_add_grad()
```

---

## Step 8 — Study Parameter Updates

Finally, follow how the calculated gradients are used to modify model parameters.

Conceptually:

```text
parameter = parameter - learning_rate × gradient
```

Repeated application of this process allows the model to minimize its loss.

---

# Example Computational Graph

A simple neural-network classifier can be represented conceptually as:

```text
                    Input
                      │
                      ▼
                ┌───────────┐
                │  MatMul   │
                │ X × W1    │
                └─────┬─────┘
                      │
                      ▼
                   ReLU
                      │
                      ▼
                ┌───────────┐
                │  MatMul   │
                │ H × W2    │
                └─────┬─────┘
                      │
                      ▼
                  Softmax
                      │
                      ▼
                Prediction
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
          Target            Prediction
             │                 │
             └────────┬────────┘
                      ▼
               Cross Entropy
                      │
                      ▼
                     Loss
```

Backpropagation then traverses this graph in the opposite direction.

---

# Design Principles

The project deliberately follows several principles.

## Minimal dependencies

The model implementation does not depend on a high-level machine-learning library.

## Explicit computation

Matrix operations are implemented directly rather than delegated to an external tensor framework.

## Explicit memory management

The arena allocator provides direct control over allocation and object lifetime.

## Explicit computational graph

Operations are represented as model variables and graph nodes rather than hidden behind a library API.

## Learning through implementation

The project prioritizes understanding the mechanisms behind ML frameworks over providing a large feature set.

---

# What This Project Demonstrates

The project demonstrates that a basic neural-network framework can be constructed from relatively small building blocks:

```text
Primitive Types
      │
      ▼
Memory Management
      │
      ▼
Matrices
      │
      ▼
Numerical Operations
      │
      ▼
Neural Network Operations
      │
      ▼
Computational Graph
      │
      ▼
Automatic Differentiation
      │
      ▼
Training
```

This hierarchy is important.

Automatic differentiation cannot be built cleanly without understanding the computation graph.

The computation graph depends on numerical operations.

The numerical operations depend on data structures.

The data structures ultimately depend on memory management.

The project therefore provides an opportunity to study the entire stack rather than only the machine-learning layer.

---

# Current Limitations

This project is intentionally small and experimental.

It currently does not attempt to provide the feature set of a production ML framework.

Some limitations include:

* Limited number of tensor operations
* Matrix-oriented rather than general N-dimensional tensor abstraction
* Limited neural-network layer support
* No GPU backend
* No SIMD/vectorized backend
* No distributed training
* Limited optimizer support
* Limited error handling
* No comprehensive unit-test suite
* No formal package/build system
* Platform-specific memory-management code
* Limited model serialization
* Limited dataset support

---

# Future Improvements

Potential extensions include:

## Tensor abstraction

Replace the 2D matrix abstraction with a general tensor abstraction supporting arbitrary dimensions.

```text
Tensor
 ├── shape
 ├── strides
 └── data
```

---

## More layers

Possible additions:

* Sigmoid
* Tanh
* Leaky ReLU
* Convolution
* Max pooling
* Batch normalization
* Dropout

---

## Optimizers

Implement:

* SGD
* Momentum
* RMSProp
* Adam

---

## Better automatic differentiation

The computational graph could be extended with:

* More operations
* Dynamic graph construction
* Gradient checking
* Graph optimization
* Gradient accumulation
* Graph visualization

---

## Performance

Potential optimizations include:

* SIMD
* Multithreading
* Cache-aware matrix multiplication
* BLAS integration
* Memory-layout optimization
* Parallel training

---

## GPU Support

A future backend could use:

* CUDA
* OpenCL
* Vulkan compute
* Metal

while preserving the same high-level computational-graph abstraction.

---

# Learning Resources

This project is particularly useful alongside topics such as:

* Linear algebra
* Matrix multiplication
* Neural networks
* Backpropagation
* Automatic differentiation
* Computational graphs
* Numerical stability
* Memory management
* C pointers
* Dynamic memory allocation
* CPU cache behavior
* Systems programming

A useful exercise is to implement one operation at a time and verify its gradient numerically using finite differences.

---

# License

The project currently does not contain a project-level `LICENSE` file.

The PCG random-number-generator component includes its own attribution and licensing information.

If the repository is intended for external reuse or distribution, a project-level license should be added.

---

# Author

**Vedansh Shrivastava**

GitHub:
https://github.com/Vedansh-Shrivastava

Project:
https://github.com/Vedansh-Shrivastava/ML-Framwork-in-C-from-Scratch

---

# Project Goal

The ultimate goal of this project is not to recreate PyTorch or TensorFlow feature-for-feature.

It is to understand the fundamental question:

> **What actually happens inside a machine-learning framework when a neural network performs a forward pass, computes a loss, calculates gradients, and updates its parameters?**

This repository attempts to answer that question by implementing those mechanisms directly in C.
