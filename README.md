# Implementation-and-Acceleration-of-a-Support-Vector-Machine-Classifier-on-the-ZCU102-FPGA-Platform

This project aims to design, implement, and evaluate a hardware-accelerated SVM inference engine on the ZCU102 board. The objective is not only functional correctness but a quantified comparison between a software (CPU) baseline and a hardware-accelerated (FPGA) implementation in terms of execution latency, throughput, resource utilization, and classification accuracy.

# SVM Algorithm Review and Design Specification

Supervised machine learning algorithms learn decision rules from labeled training samples to classify unseen inputs. Formally, consider a dataset **D = {(x₁, y₁), (x₂, y₂), …, (x_N, y_N)}**, where each **xᵢ ∈ ℝᴰ** represents a D-dimensional feature vector and **yᵢ ∈ {−1, +1}** denotes the binary class label. The objective is to construct a mapping function **f: ℝᴰ → {−1, +1}** that accurately predicts class labels for new feature vectors.

The operational flow of the Support Vector Machine inference engine processes input vectors through sequential transformation stages, as outlined below:

- **Input Sample Ingress: **Receives unscaled D-dimensional feature vectors x ∈ ℝᴰ from external sensor streams or DDR host memory buffers.
- **Feature Scaling Normalization: **Scales individual raw features linearly into a standardized range, typically [−1.0, +1.0], to prevent larger feature scales from biasing distance computations.
- **Parallel Kernel Evaluation: **Computes inner product transformations K(x_m, x) concurrently across all stored Support Vectors x_m using selected kernel functions.
- **Weighted Linear Accumulation: **Multiplies calculated kernel values by stored dual coefficients β_m = α_m y_m and sums them alongside the scalar bias offset b.
- **Class Sign Output Assignment: **Evaluates the sign of the accumulated scalar decision value to output the final binary class prediction +1 or −1.
> **SVM Inference Equation:**
>
> `f(x) = sign( Σ αᵢ yᵢ K(xᵢ, x) + b )`
><img width="433" height="103" alt="image" src="https://github.com/user-attachments/assets/3ea30112-2518-42b6-91ee-3dd3a3b73de3" />
> Where αᵢ are the dual coefficients, yᵢ are class labels, K(·,·) is the kernel function, and b is the bias offset.

## Train the SVM offline using LIBSVM

> 📥 *Download LIBSVM from [https://www.csie.ntu.edu.tw/~cjlin/libsvm/](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)*

> 📥 *Download the `a1a` dataset from the LIBSVM Data repository*

**After training:**

The SMO optimization algorithm successfully completed training after 8,649 iterations, converging to the optimal decision boundary.

rho = 1.594468 This is the offset (ρ) reported by LIBSVM b=−ρ. 

Total nSV = 591  The trained SVM model contains only 591 Support Vectors. 

> 📸 *Figure: Trained LIBSVM model output showing convergence parameters*

# CPU Baseline Implementation

The goal of this phase is to establish a software baseline before hardware acceleration. This baseline provides the reference performance against which the FPGA implementation will be compared.

A detailed timing profile was performed by separating the execution into three main stages:

Model Parsing: Reading and loading the LIBSVM model into memory. 

Dataset Parsing: Reading and preparing the input test samples. 

Pure Inference: Executing the SVM classification algorithm on the CPU. 

The timing profile clearly indicates that Pure Inference dominates the total execution time, accounting for approximately 88% of the overall pipeline. This is expected because the inference stage performs the computationally intensive SVM operations, including numerous multiply-accumulate (MAC) computations and dot-product calculations between each input sample and all support vectors.

Since the vast majority of the execution time is spent in the inference stage, it becomes the primary performance bottleneck. Therefore, the FPGA implementation focuses on accelerating only the inference engine, where the large amount of parallel computations can be efficiently mapped to hardware. The parsing stages remain on the CPU because they contribute only a small fraction of the total execution time and would provide minimal performance improvement if accelerated.

# Vitis

## Host Program

### 1. Header Guard

```
#pragma once
```

This prevents the file from being included multiple times during compilation. For example, if you #include "host_main.h" twice across different files, this guard ensures you won't trigger **Compilation Errors** (such as redefinition errors).

### 2. OpenCL Version Specification

```
#define CL_HPP_TARGET_OPENCL_VERSION 120
```

This macro instructs the compiler to target **OpenCL 1.2**. This is required because the Xilinx RunTime (XRT) inside AMD Vitis is built upon OpenCL 1.2 standards.

### 3. OpenCL C++ Header

```
#include <CL/cl2.hpp>
```

This is the primary Header file. It provides wrapper classes for managing the FPGA hardware, ou do not need to write custom kernel drivers from scratch. AMD/Xilinx provides the high-level **OpenCL C++ API** to orchestrate host-accelerator communication including:

| OpenCL Class | Functional Purpose |
| --- | --- |
| cl::Platform | Discovers vendor drivers (e.g., verifying Xilinx target platforms). |
| cl::Device | Identifies connected FPGA target hardware (e.g., AMD ZCU102 or Alveo cards). |
| cl::Context | Creates a shared execution session connecting Host CPU memory and FPGA resources. |
| cl::CommandQueue | Serves as an in-order command stream (memory transfers, kernel executions). |
| cl::Program | Loads the compiled .xclbin bitstream into host runtime memory. |
| cl::Kernel | Binds a callable handle to the hardware accelerator function inside the FPGA. |
| cl::Buffer | Allocates and references dedicated memory regions inside off-chip DDR memory. |

Without this library, you cannot interface with or control the FPGA hardware from host code.

### 4. aligned_allocator (Custom Memory Allocation)

```
template<typename T>struct aligned_allocator
```

This concept often confuses developers. Why do we need custom aligned memory allocation?

Consider a standard host array like std::vector<float>. The default dynamic allocator might place memory at arbitrary addresses, such as 0x1003 or 0x2037.

However, the Direct Memory Access (**DMA**) engine on the FPGA requires memory addresses to align to **4096-Byte boundaries** (4 KB pages):

```
Ideal DMA Alignment:      0x1000, 0x2000, 0x3000
Unaligned (Sub-optimal):   0x2003, 0x1003
```

By wrapping posix_memalign(&ptr, 4096, ...) inside a custom vector allocator, the starting address of host memory is guaranteed to be a multiple of 4096. This eliminates alignment overhead and significantly improves DMA transfer performance.

## Part 2: Main Application Workflow (main())

### 1. Path Resolution

```
getcwd(...)
```

Prints the current working directory. This helps confirm the operational path at runtime and resolves file-path issues (such as Unable to open model).

### 2. File Artifacts

- binary_container_1.xclbin: The compiled FPGA **Bitstream** binary generated by Vitis.
- a1a.model: The trained model parameters file.
- a1a.t: The input test dataset.

### 3. Data Parsing (Software Phase)

```
parse_libsvm_model(...);
parse_libsvm_dataset(...);
```

During this initial phase, the **CPU alone** reads and parses input files into memory. The FPGA is not yet active—this entire stage runs purely in software on the Host processor.

***flatten vector<vector<float>> into a 1D vector***

```
host_sv_vectors[m * num_features + f]
```

The outer vector does not store actual values—it holds **pointers** to individual inner vectors allocated dynamically across different RAM locations:

#### Why is this layout bad for FPGA accelerators?

**FPGA hardware does not understand Standard Template Libraries (STL)** or nested pointers.

The FPGA DMA engine expects contiguous physical memory addresses:

so We convert the 2D layout into a contiguous 1D host array (host_sv_vectors)

```
host_sv_vectors[m * num_features + f]
```

DDR memory operates most efficiently using **Burst Reads**. Instead of triggering separate non-contiguous read requests across 0x1000, 0x5300, and 0xAB00, a flattened layout allows the AXI Master to issue a single streaming Burst Read starting at address 0x1000.

```
cl::Kernel krnl_svm(program, "svm_kernel_accel")
```

An .xclbin container can include multiple hardware kernels (kernel0, kernel1, kernel2).

Loading the bitstream via cl::Program program(...) transfers the compiled hardware image into the execution environment. Writing:

```
cl::Kernel krnl_svm(program, "svm_kernel_accel");
```

Explicitly queries the loaded program and establishes a software-side reference handle to the hardware kernel symbol named "svm_kernel_accel". Once bound, you can invoke API calls such as .setArg() and enqueueTask().

**cl::Buffer**

 is **not** memory itself; it is an abstract **Object Reference** representing an allocated region inside board-level DDR Memory.

```
Global DDR Memory Space
┌──────────────────────────────────┐
│ Buffer: Samples                  │
├──────────────────────────────────┤
│ Buffer: Support Vectors          │
├──────────────────────────────────┤
│ Buffer: Coefficients             │
├──────────────────────────────────┤
│ Buffer: Predictions              │
└──────────────────────────────────┘
```

When you construct cl::Buffer buffer_samples(...), OpenCL registers an address range within global DDR memory.

Data remains in host system RAM (host_samples) until explicitly transferred to these DDR buffer locations using enqueueMigrateMemObjects().

**CL_MEM_USE_HOST_PTR**

By supplying CL_MEM_USE_HOST_PTR, you instruct the XRT runtime to map the memory pointer (host_samples.data()) directly to the OpenCL buffer instance, eliminating unnecessary intermediate host-RAM memory copies.

**Note:** This does not mean the FPGA accesses CPU system RAM directly. The FPGA still reads from off-chip **DDR memory** over the PCIe/AXI bus; CL_MEM_USE_HOST_PTR simply eliminates redundant copies on the host CPU side before DMA migration takes place.

> 📸 *Figure: System execution order — Data parsing (CPU) → Buffer transfer → Kernel execution (FPGA) → Result readback (CPU)*

## Kernal

### HLS INTERFACE

This is the most critical part of the kernel architecture.

#### 1. m_axi (AXI Master)

For example:

```
#pragma HLS INTERFACE m_axi port=in_samples bundle=gmem0
```

This represents the data flow:

The kernel directly reads data from DDR memory on its own.

**Why is it called ***Master***?**

Because the **kernel itself initiates** the read/write memory operations (rather than waiting for an external controller).

#### 2. bundle

For example: gmem0

All ports assigned the same bundle name share the same physical AXI port.

In your design:

- Samples = gmem0
- Predictions = gmem0
While:

- Support Vectors = gmem1
This gives Support Vectors a **dedicated, independent memory interface**, which reduces memory contention and port bottlenecks.

#### 3. depth

```
depth=3807588
```

This represents the **maximum expected number of elements**.

It does *not* allocate memory itself; rather, it helps the HLS tool estimate interface sizes and optimize hardware during synthesis and simulation

#### 4. max_read_burst_length

```
max_read_burst_length=256
```

Instead of performing individual reads cycle-by-cycle:

It performs a **Burst Read** of up to **256 values at once**, significantly boosting memory bandwidth and throughput.

#### 5. s_axilite (AXI-Lite)

```
#pragma HLS INTERFACE s_axilite
```

These are **Control Registers**.

The CPU uses this interface to write parameters such as num_samples, num_sv, and rho into control registers before starting the kernel.

The control flow looks like this:

#### 6. local_sv (Local On-Chip Buffer)

```
custom_data_t local_sv[MAX_SV][MAX_FEATURES];
```

This is the **most important array** in the entire codebase.

**Why?**

Instead of reading from DDR memory repeatedly during execution, it transfers the Support Vectors **once** from global memory into fast on-chip memory (Block RAM / URAM):

This dramatically reduces memory latency and accelerates computation.

### Optimization in Kernal

#### ARRAY_PARTITION

```
#pragma HLS ARRAY_PARTITION
```

This is one of the most critical optimizations in HLS.

If you have data stored in memory:

#### Without Partitioning

A standard BRAM port can **only read 1 element per clock cycle**.

#### After applying factor=8

The memory is split into separate banks:

Now, the hardware can **read all 8 Features simultaneously in the exact same clock cycle**.

This is precisely why the **UNROLL factor** works—unrolling hardware logic only achieves true parallelism if memory bandwidth can feed all parallel units at once!

#### Using Paragms such as PIPELINE = Parallelism Across Iterations

#### What is an Iteration?

Consider this loop:

This loop will execute **4 times**.

#### Without Pipelining

Let's assume that each iteration requires **3 Clock cycles** to complete.

The execution will look like this:

#### With Pipelining (II = 1)

Here, HLS (High-Level Synthesis) says:

*"Instead of waiting for an iteration to finish, I will start a new one every single Clock cycle."*

The execution becomes:

**Notice what happened:** At **Clock 3**, you have:

Iteration 0

Iteration 1

Iteration 2

All of them are running **at the same time**, but each one is at a **different stage** of execution.

#### So, what does PIPELINING actually do?

It does **not** reduce the time taken by a single iteration.

Instead, it increases the number of iterations running concurrently.

In other words, it increases **Throughput**, not **Latency**.

#### UNROLL

unroll= Parallelism Inside One Iteration

Consider this loop:

We have **8 Features**.

#### Without UNROLL

There is **only 1 Multiplier**.

So it executes sequentially like this:

In other words, it uses the **exact same multiplier 8 times in a row**.

#### With #pragma HLS UNROLL factor=8

HLS says:

*"Instead of using 1 multiplier... I will build **8 multipliers** in hardware."*

The circuit becomes:

They **all execute in the very same Clock cycle**:

#### So, what does UNROLL actually do?

It decreases the time needed to finish the loop by executing the operations inside it **in parallel**, replicating hardware resources to achieve this speedup.

### In your specific code, this happens because:

The **outer loop** (EVAL_SV_LOOP) has **PIPELINE** applied to it.

The **inner loop** (DOT_PRODUCT_LOOP) has **UNROLL** applied to it.

Let’s apply this directly to our code.

**Let's assume:**

- Number of Support Vectors (num_sv) = **3**
- Number of Features (MAX_FEATURES) = **8**

#### First: UNROLL

#### Without UNROLL (for SV0):

*This means computing a single dot product takes **8 clock cycles**.*

#### With #pragma HLS UNROLL factor=8:

*All multiplications are executed together.*

**UNROLL** sped up the calculation of a single dot product.

#### Second: PIPELINE

Now that computing each Support Vector is fast, **PIPELINE** comes into play.

#### Without PIPELINE:

*SV1** does not start until **SV0** is completely finished.*

#### With #pragma HLS PIPELINE II=1:

**Notice:**

- SV0 is still executing.
- SV1 has already started.
- SV2 has started as well.
**PIPELINE** allows processing of a **new Support Vector** to start **every clock cycle**.

#### Combining Both Together

Now picture it like this:

> 📸 *Figure: Software emulation result on Vitis showing correctness of kernel output before hardware deployment*

## Host Integration and Functional Testing

The synthesized kernel is integrated with a host application and deployed to the ZCU102 board using the Vitis embedded platform flow.

In this final phase, the complete acceleration framework was deployed onto the physical **Xilinx Zynq UltraScale+ MPSoC ZCU102 Evaluation Kit** using **SD Card Boot Mode**, transitioning away from interactive JTAG debugging to a standalone Embedded Linux environment.

### 1. SD Card Preparation and Partitioning

Using the **GParted** disk management utility, the SD card was formatted and split into two distinct primary partitions:

- **boot** Partition (FAT32):** Contains the essential bootloader components and runtime deployment artifacts generated by Vitis:
- BOOT.BIN *(First Stage Bootloader / FSBL, PMU Firmware, and U-Boot)*
- boot.scr *(U-Boot configuration script)*
- Image *(Linux Kernel Image)*
- binary_container_1.xclbin *(FPGA Hardware Bitstream)*
- svm_model *(Compiled C++ Host Executable)*
- Model & Dataset files: a1a.model and a1a.t

- **rootfs** Partition (ext4):** Extracted from the Xilinx Linux Common Image (rootfs.tar.gz), hosting the complete Linux target root filesystem, standard libraries, and XRT (Xilinx Runtime) environment.

### 2. Board Configuration and Hardware Setup

To configure the board for SD Card booting rather than JTAG:

- **DIP Switch Settings (SW6):** The boot mode switches were set to **E-MODE / SD Card Boot** by configuring the DIP switches to **0111** (specifically setting SW6 pins [1:4] to ON-OFF-OFF-OFF).
- **UART Serial Interface:** Connected the host PC to the board via USB (UART USB0 port). A serial terminal emulator (**GtkTerm**) was launched on the host machine using a standard baud rate of **115200** baud to monitor the Embedded Linux boot sequence and interact with the kernel.

### 3. Execution & Verification

Once Embedded Linux booted successfully, the host executable was launched via the terminal to evaluate the SVM acceleration:

```
./svm_model binary_container_1.xclbin a1a.model a1a.t
```

The execution verified that the hardware bitstream was successfully loaded into the Programmable Logic (PL) via XRT, completing the SVM inference on the acceleration platform and displaying real-time benchmark results over UART.

# Hardware Results & Inference Speedup

"Upon deploying the design onto the physical hardware (ZCU102 board), the inference latency was significantly reduced from **56 ms** (software baseline) down to **12 ms**, achieving a ****4.67×** speedup** and demonstrating the efficiency of our FPGA hardware acceleration architecture.

Performance Comparison: CPU Baseline vs. Software Emulation vs. FPGA Acceleration

| Metric | CPU Baseline | Software Emulation | FPGA Acceleration |
| --- | --- | --- | --- |
| Total Inference Time | 56.326 ms | ~Simulation | **12.249 ms** |
| Throughput | 8,876.88 samples/sec | — | **40,820.89 samples/sec** |
| Speedup | 1× (baseline) | — | **~4.6× faster** |
