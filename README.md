# Implementation-and-Acceleration-of-a-Support-Vector-Machine-Classifier-on-the-ZCU102-FPGA-Platform

This project aims to design, implement, and evaluate a hardware-accelerated SVM inference engine on the ZCU102 board. The objective is not only functional correctness but a quantified comparison between a software (CPU) baseline and a hardware-accelerated (FPGA) implementation in terms of execution latency, throughput and classification accuracy.

# SVM Algorithm Review and Design Specification
In Support Vector Machines, this boundary line is called a hyperplane. The boundary lines parallel to the hyperplane that touch the closest sample points define the margin. The specific sample points touching these boundary lines are called Support Vectors. 
The goal is to draw a straight line that cleanly divides the blue triangles from the green circles. While many lines could separate these groups, an optimal line leaves the widest possible margin on both sides.<img width="433" height="103" alt="image" src="Images/SVM1.png" />
> **SVM Inference Equation:**
>
<img width="433" height="103" alt="image" src="Images/Picture1.png" />
> Where αᵢ are the dual coefficients, yᵢ are class labels, K(·,·) is the kernel function, and b is the bias offset.
Analyzed SVM model's operational flow by deeply understanding its core inference equation. As illustrated in the figure, we mapped this mathematical formula into a sequential, five-step execution pipeline to clearly define the exact step-by-step calculations required for the final classification decision.
><img width="433" height="103" alt="image" src="Images/SVM2.png" />
> 
## Train the SVM offline using LIBSVM
completed training after 8,649 iterations
# Implementation-and-Acceleration-of-a-Support-Vector-Machine-Classifier-on-the-ZCU102-FPGA-Platform

<style>
	/* Inline images are inline-block to avoid float overlap; small images stay next to text */
	.inline-img { display: inline-block; width: 300px; margin: 0 12px 12px 12px; vertical-align: middle; height: auto; }
	/* Block images are centered and have spacing so following text won't overlap */
	.block-img { display: block; margin: 18px auto; max-width: 900px; width: 100%; height: auto; }
	/* Ensure headings and paragraphs sit below images if needed */
	.clear { clear: both; display: block; height: 0; }
	@media (max-width: 800px) { .inline-img { display: block; margin: 8px auto; } }
</style>

This project aims to design, implement, and evaluate a hardware-accelerated SVM inference engine on the ZCU102 board. The objective is not only functional correctness but a quantified comparison between a software (CPU) baseline and a hardware-accelerated (FPGA) implementation in terms of execution latency, throughput and classification accuracy.

# SVM Algorithm Review and Design Specification
In Support Vector Machines, this boundary line is called a hyperplane. The boundary lines parallel to the hyperplane that touch the closest sample points define the margin. The specific sample points touching these boundary lines are called Support Vectors. 
The goal is to draw a straight line that cleanly divides the blue triangles from the green circles. While many lines could separate these groups, an optimal line leaves the widest possible margin on both sides. <img class="inline-img" alt="SVM1" src="Images/SVM1.png" />

> **SVM Inference Equation:**
>
> <img class="block-img" alt="Picture1" src="Images/Picture1.png" />
> Where αᵢ are the dual coefficients, yᵢ are class labels, K(·,·) is the kernel function, and b is the bias offset.
Analyzed SVM model's operational flow by deeply understanding its core inference equation. As illustrated in the figure, we mapped this mathematical formula into a sequential, five-step execution pipeline to clearly define the exact step-by-step calculations required for the final classification decision.

> <img class="block-img" alt="SVM2" src="Images/SVM2.jpeg" />

## Train the SVM offline using LIBSVM
completed training after 8,649 iterations
properities 
Kernal type : linear 
num. of support vectors : 591
num. of features = 123
num. of dual coff : 591
rho : 1.594468

<img class="block-img" alt="SVM3" src="Images/SVM3.png" />

# Time Profiling & System Partitioning

<img class="block-img" alt="SVM8" src="Images/SVM8.png" />

<img class="block-img" alt="SVM9" src="Images/SVM9.png" />

Time Profiling:
 Evaluates CPU baseline execution to analyze time distribution per function.
 Locating Computational Bottlenecks: Identifies time-consuming functions (e.g., matrix operations & loops) for acceleration 
System Partitioning:
 Software (CPU): Control logic, File I/O, data parsing, and pre/post-processing. 
 Hardware (FPGA): Compute-heavy mathematical kernels targeting acceleration. 
## CPU Baseline Implementation

<img class="block-img" alt="SVM4" src="Images/SVM4.png" />

The goal of this phase is to establish a software baseline before hardware acceleration. This baseline provides the reference performance against which the FPGA implementation will be compared.

A detailed timing profile was performed by separating the execution into three main stages:

Model Parsing: Reading and loading the LIBSVM model into memory. 

Dataset Parsing: Reading and preparing the input test samples. 

Pure Inference: Executing the SVM classification algorithm on the CPU. 

The timing profile clearly indicates that Pure Inference dominates the total execution time, accounting for approximately 88% of the overall pipeline. This is expected because the inference stage performs the computationally intensive SVM operations, including numerous multiply-accumulate (MAC) computations and dot-product calculations between each input sample and all support vectors.

<img class="block-img" alt="SVM5" src="Images/SVM5.png" />

Since the vast majority of the execution time is spent in the inference stage, it becomes the primary performance bottleneck. Therefore, the FPGA implementation focuses on accelerating only the inference engine, where the large amount of parallel computations can be efficiently mapped to hardware. The parsing stages remain on the CPU because they contribute only a small fraction of the total execution time and would provide minimal performance improvement if accelerated.

# Vitis

## Host Program
1. Environment Setup & Initialization
The host program begins by setting up the OpenCL environment (targeting OpenCL 1.2 for Xilinx XRT compatibility). It includes the <CL/cl2.hpp> headers to create the necessary software objects (Platform, Device, Context, and CommandQueue) that will orchestrate communication between the CPU and the FPGA.

2. Data Loading & Parsing (Software Phase)
Running purely on the CPU, the program reads the necessary file artifacts from the current directory. It parses the trained model (a1a.model) and the test dataset (a1a.t) into system RAM, while simultaneously preparing to load the compiled FPGA bitstream (.xclbin).

3. Memory Alignment & Data Flattening
To ensure the FPGA's Direct Memory Access (DMA) engine can read data efficiently, the host code optimizes the data layout:

Flattening: It converts nested 2D data structures (which are scattered in memory) into contiguous 1D vectors. This allows the hardware to use fast, streaming "Burst Reads."

Alignment: It uses a custom aligned_allocator to align the starting addresses of this memory to 4096-Byte (4 KB) boundaries, preventing alignment overhead during data transfers.

4. FPGA Configuration & Hardware Binding
The host loads the .xclbin bitstream into the runtime memory to configure the FPGA. It then creates a cl::Kernel object to bind a software handle to the specific hardware function (e.g., "svm_kernel_accel").

5. Buffer Allocation
The host allocates cl::Buffer objects. These act as references to designated memory regions inside the FPGA's off-chip DDR memory. By using the CL_MEM_USE_HOST_PTR flag, the host maps the OpenCL buffers directly to the aligned CPU memory pointers, avoiding redundant intermediate memory copies on the CPU side.

6. Host to FPGA Transfer
The CPU copies the flattened, aligned dataset and model parameters from the Host RAM over the PCIe bus into the FPGA's global DDR memory.

7. Kernel Launch & Execution
The host issues a command to launch the kernel. The FPGA takes over and executes the hardware-accelerated function (the SVM computations) using the data now residing in its DDR memory.

8. Results Transfer & Verification
Once the FPGA finishes computing, the host reads the output by copying the results from the FPGA's DDR memory back to the CPU memory. Finally, the host program verifies the accuracy of the predictions and cleans up the execution environment.

<img class="block-img" alt="SVM6" src="Images/SVM6.png" />

## Vitis Kernal
Definition: An isolated, compute-intensive function (C/C++) compiled into dedicated FPGA hardware logic.
HLS Directives (Pragmas): Guiding the Vitis HLS compiler to optimize hardware architecture and parallel execution.

<img class="block-img" alt="SVM7" src="Images/SVM7.png" />

### Optimization in Kernal

#### ARRAY_PARTITION

```
#pragma HLS ARRAY_PARTITION
```
This is one of the most critical optimizations in HLS.

If you have data stored in memory:

<img class="block-img" alt="SVM10" src="Images/SVM10.png" />

#### Without Partitioning

A standard BRAM port can **only read 1 element per clock cycle**.

#### After applying factor=8

The memory is split into separate banks:
<img class="block-img" alt="SVM11" src="Images/SVM11.png" />

Now, the hardware can **read all 8 Features simultaneously in the exact same Clock cycle**.

This is precisely why the **UNROLL factor** works—unrolling hardware logic only achieves true parallelism if memory bandwidth can feed all parallel units at once!

#### Using Paragms such as PIPELINE = Parallelism Across Iterations

#### What is an Iteration?

Consider this loop:
<img class="block-img" alt="SVM12" src="Images/SVM12.png" />

This loop will execute **4 times**.
<img class="block-img" alt="SVM13" src="Images/SVM13.png" />

#### Without Pipelining

Let's assume that each iteration requires **3 Clock cycles** to complete.

The execution will look like this:
<img class="block-img" alt="SVM14" src="Images/SVM14.png" />


#### With Pipelining (II = 1)

Here, HLS (High-Level Synthesis) says:

"*Instead of waiting for an iteration to finish, I will start a new one every single Clock cycle.*"

The execution becomes:
<img class="block-img" alt="SVM15" src="Images/SVM15.png" />


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
<img class="block-img" alt="SVM16" src="Images/SVM16.png" />


We have **8 Features**.

#### Without UNROLL

There is **only 1 Multiplier**.

So it executes sequentially like this:
<img class="block-img" alt="SVM17" src="Images/SVM17.png" />


In other words, it uses the **exact same multiplier 8 times in a row**.

#### With #pragma HLS UNROLL factor=8

HLS says:

"*Instead of using 1 multiplier... I will build **8 multipliers** in hardware.*"

The circuit becomes:
<img class="block-img" alt="SVM18" src="Images/SVM18.png" />


They **all execute in the very same Clock cycle**:

#### So, what does UNROLL actually do?

It decreases the time needed to finish the loop by executing the operations inside it **in parallel**, replicating hardware resources to achieve this speedup.

### In your specific code, this happens because:

The **outer loop** (EVAL_SV_LOOP) has **PIPELINE** applied to it.

The **inner loop** (DOT_PRODUCT_LOOP) has **UNROLL** applied to it.

Let’s apply this directly to our code.
<img class="block-img" alt="SVM19" src="Images/SVM19.png" />


**Let's assume:**

- Number of Support Vectors (num_sv) = **3**
- Number of Features (MAX_FEATURES) = **8**

#### First: UNROLL

#### Without UNROLL (for SV0):
<img class="block-img" alt="SVM20" src="Images/SVM20.png" />

*This means computing a single dot product takes **8 clock cycles**.*

#### With #pragma HLS UNROLL factor=8:
<img class="block-img" alt="SVM21" src="Images/SVM21.png" />


*All multiplications are executed together.*

**UNROLL** sped up the calculation of a single dot product.

#### Second: PIPELINE

Now that computing each Support Vector is fast, **PIPELINE** comes into play.

#### Without PIPELINE:
<img class="block-img" alt="SVM22" src="Images/SVM22.png" />


*SV1** does not start until **SV0** is completely finished.*

#### With #pragma HLS PIPELINE II=1:
<img class="block-img" alt="SVM23" src="Images/SVM23.png" />


**Notice:**

- SV0 is still executing.
- SV1 has already started.
- SV2 has started as well.
**PIPELINE** allows processing of a **new Support Vector** to start **every clock cycle**.

#### Combining Both Together

Now picture it like this:

<img class="block-img" alt="SVM24" src="Images/SVM24.png" />

## Software Emulation Result on Vitis
Software Emulation in the Vitis IDE was utilized to verify host-kernel communication and functional correctness before initiating the time-consuming hardware build. Although virtualization overhead yields performance metrics worse than the CPU baseline, this phase is strictly for rapid bug detection and functional validation prior to physical FPGA deployment.
<img class="block-img" alt="SVM25" src="Images/SVM25.png" />

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
- <img class="block-img" alt="SVM26" src="Images/SVM26.png" />


- **rootfs** Partition (ext4):** Extracted from the Xilinx Linux Common Image (rootfs.tar.gz), hosting the complete Linux target root filesystem, standard libraries, and XRT (Xilinx Runtime) environment.
- <img class="block-img" alt="SVM27" src="Images/SVM27.png" />


### 2. Board Configuration and Hardware Setup

To configure the board for SD Card booting rather than JTAG:

- **DIP Switch Settings (SW6):** The boot mode switches were set to **E-MODE / SD Card Boot** by configuring the DIP switches to **0111** (specifically setting SW6 pins [1:4] to ON-OFF-OFF-OFF).
- **UART Serial Interface:** Connected the host PC to the board via USB (UART USB0 port). A serial terminal emulator (**GtkTerm**) was launched on the host machine using a standard baud rate of **115200** baud to monitor the Embedded Linux boot sequence and interact with the kernel.
- <img class="block-img" alt="SVM28" src="Images/SVM28.png" />


### 3. Execution & Verification

Once Embedded Linux booted successfully, the host executable was launched via the terminal to evaluate the SVM acceleration:

```
./svm_model binary_container_1.xclbin a1a.model a1a.t
```

The execution verified that the hardware bitstream was successfully loaded into the Programmable Logic (PL) via XRT, completing the SVM inference on the acceleration platform and displaying real-time benchmark results over UART.

# Hardware Results & Inference Speedup

## FPGA Acceleration

<img class="block-img" alt="SVM29" src="Images/SVM29.png" />

"Upon deploying the design onto the physical hardware (ZCU102 board), the inference latency was significantly reduced from **56 ms** (software baseline) down to **12 ms**, achieving a ****4.67×** speedup** and demonstrating the efficiency of our FPGA hardware acceleration architecture.

<img class="block-img" alt="SVM30" src="Images/SVM30.png" />


| Throughput | 8,876.88 samples/sec | — | **40,820.89 samples/sec** |
| Speedup | 1× (baseline) | — | **~4.6× faster** |


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
<img class="block-img" alt="SVM16" src="Images/SVM16.png" />


We have **8 Features**.

#### Without UNROLL

There is **only 1 Multiplier**.

So it executes sequentially like this:
<img class="block-img" alt="SVM17" src="Images/SVM17.png" />


In other words, it uses the **exact same multiplier 8 times in a row**.

#### With #pragma HLS UNROLL factor=8

HLS says:

"*Instead of using 1 multiplier... I will build **8 multipliers** in hardware.*"

The circuit becomes:
<img class="block-img" alt="SVM18" src="Images/SVM18.png" />


They **all execute in the very same Clock cycle**:

#### So, what does UNROLL actually do?

It decreases the time needed to finish the loop by executing the operations inside it **in parallel**, replicating hardware resources to achieve this speedup.

### In your specific code, this happens because:

The **outer loop** (EVAL_SV_LOOP) has **PIPELINE** applied to it.

The **inner loop** (DOT_PRODUCT_LOOP) has **UNROLL** applied to it.

Let’s apply this directly to our code.
<img class="block-img" alt="SVM19" src="Images/SVM19.png" />


**Let's assume:**

- Number of Support Vectors (num_sv) = **3**
- Number of Features (MAX_FEATURES) = **8**

#### First: UNROLL

#### Without UNROLL (for SV0):
<img class="block-img" alt="SVM20" src="Images/SVM20.png" />

*This means computing a single dot product takes **8 clock cycles**.*

#### With #pragma HLS UNROLL factor=8:
<img class="block-img" alt="SVM21" src="Images/SVM21.png" />


*All multiplications are executed together.*

**UNROLL** sped up the calculation of a single dot product.

#### Second: PIPELINE

Now that computing each Support Vector is fast, **PIPELINE** comes into play.

#### Without PIPELINE:
<img class="block-img" alt="SVM22" src="Images/SVM22.png" />


*SV1** does not start until **SV0** is completely finished.*

#### With #pragma HLS PIPELINE II=1:
<img class="block-img" alt="SVM23" src="Images/SVM23.png" />


**Notice:**

- SV0 is still executing.
- SV1 has already started.
- SV2 has started as well.
**PIPELINE** allows processing of a **new Support Vector** to start **every clock cycle**.

#### Combining Both Together

Now picture it like this:

<img class="block-img" alt="SVM24" src="Images/SVM24.png" />

## Software Emulation Result on Vitis
Software Emulation in the Vitis IDE was utilized to verify host-kernel communication and functional correctness before initiating the time-consuming hardware build. Although virtualization overhead yields performance metrics worse than the CPU baseline, this phase is strictly for rapid bug detection and functional validation prior to physical FPGA deployment.
<img class="block-img" alt="SVM25" src="Images/SVM25.png" />

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
- <img class="block-img" alt="SVM26" src="Images/SVM26.png" />


- **rootfs** Partition (ext4):** Extracted from the Xilinx Linux Common Image (rootfs.tar.gz), hosting the complete Linux target root filesystem, standard libraries, and XRT (Xilinx Runtime) environment.
- <img class="block-img" alt="SVM27" src="Images/SVM27.png" />


### 2. Board Configuration and Hardware Setup

To configure the board for SD Card booting rather than JTAG:

- **DIP Switch Settings (SW6):** The boot mode switches were set to **E-MODE / SD Card Boot** by configuring the DIP switches to **0111** (specifically setting SW6 pins [1:4] to ON-OFF-OFF-OFF).
- **UART Serial Interface:** Connected the host PC to the board via USB (UART USB0 port). A serial terminal emulator (**GtkTerm**) was launched on the host machine using a standard baud rate of **115200** baud to monitor the Embedded Linux boot sequence and interact with the kernel.
- <img class="block-img" alt="SVM28" src="Images/SVM28.png" />


### 3. Execution & Verification

Once Embedded Linux booted successfully, the host executable was launched via the terminal to evaluate the SVM acceleration:

```
./svm_model binary_container_1.xclbin a1a.model a1a.t
```

The execution verified that the hardware bitstream was successfully loaded into the Programmable Logic (PL) via XRT, completing the SVM inference on the acceleration platform and displaying real-time benchmark results over UART.

# Hardware Results & Inference Speedup

## FPGA Acceleration
 
<img class="block-img" alt="SVM29" src="Images/SVM29.png" />

"Upon deploying the design onto the physical hardware (ZCU102 board), the inference latency was significantly reduced from **56 ms** (software baseline) down to **12 ms**, achieving a ****4.67×** speedup** and demonstrating the efficiency of our FPGA hardware acceleration architecture.

<img class="block-img" alt="SVM30" src="Images/SVM30.png" />


| Throughput | 8,876.88 samples/sec | — | **40,820.89 samples/sec** |
| Speedup | 1× (baseline) | — | **~4.6× faster** |
