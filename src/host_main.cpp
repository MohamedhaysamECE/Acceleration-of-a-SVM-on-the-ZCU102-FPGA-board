#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include <limits.h>
#include "host_main.h"
#include "libsvm_parser.h"

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

typedef float custom_data_t;

int main(int argc, char** argv) {
	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
	    std::cout << "Current path = " << cwd << std::endl;
	}
    std::string xclbinFilename = "binary_container_1.xclbin";
    
    std::string model_path = "data/a1a.model";
    std::string dataset_path = "data/a1a.t";
    int num_features = 123;

    if (argc >= 4) {
        xclbinFilename = argv[1];
        model_path = argv[2];
        dataset_path = argv[3];
    }

    std::cout << "========== SVM Hardware Acceleration (Native OpenCL) ==========" << std::endl;

    // --- 1. Software Phase: Model Parsing ---
    auto t_model_start = std::chrono::high_resolution_clock::now();
    SVMModel model = parse_libsvm_model(model_path, num_features);
    auto t_model_end = std::chrono::high_resolution_clock::now();
    double model_load_time_ms = std::chrono::duration<double, std::milli>(t_model_end - t_model_start).count();

    // --- 2. Software Phase: Dataset Parsing ---
    auto t_dataset_start = std::chrono::high_resolution_clock::now();
    Dataset dataset = parse_libsvm_dataset(dataset_path, num_features);
    auto t_dataset_end = std::chrono::high_resolution_clock::now();
    double dataset_load_time_ms = std::chrono::duration<double, std::milli>(t_dataset_end - t_dataset_start).count();


    std::cout << "Data Parsed Successfully by Software:" << std::endl;
    std::cout << " - Support Vectors Count: " << model.total_sv << std::endl;
    std::cout << " - Total Test Samples   : " << dataset.num_samples << std::endl;

    std::vector<custom_data_t, aligned_allocator<custom_data_t>> host_samples(dataset.num_samples * num_features);
    std::vector<custom_data_t, aligned_allocator<custom_data_t>> host_sv_vectors(model.total_sv * num_features);
    std::vector<custom_data_t, aligned_allocator<custom_data_t>> host_dual_coeffs(model.total_sv);
    std::vector<custom_data_t, aligned_allocator<custom_data_t>> host_predictions(dataset.num_samples);

    for (int s = 0; s < dataset.num_samples; ++s) {
        for (int f = 0; f < num_features; ++f) {
            host_samples[s * num_features + f] = (custom_data_t)dataset.features[s][f];
        }
    }

    for (int m = 0; m < model.total_sv; ++m) {
        host_dual_coeffs[m] = (custom_data_t)model.dual_coeffs[m];
        for (int f = 0; f < num_features; ++f) {
            host_sv_vectors[m * num_features + f] = (custom_data_t)model.support_vectors[m][f];
        }
    }

    // --- 4. OpenCL Native Environment Setup ---
    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_svm;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) && (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (!found_device) {
        std::cout << "Error: Unable to find Target Xilinx Device" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    if (!bin_file.is_open()) {
        std::cout << "ERROR: " << xclbinFilename << " file not available!" << std::endl;
        return EXIT_FAILURE;
    }
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        
        std::cout << "Programming device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            
            OCL_CHECK(err, krnl_svm = cl::Kernel(program, "svm_kernel_accel", &err));
            valid_device = true;
            break;
        }
    }
    delete[] buf;

    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        return EXIT_FAILURE;
    }

    // --- 5. Buffer Allocation using CL_MEM_USE_HOST_PTR  ---
    size_t samples_bytes = host_samples.size() * sizeof(custom_data_t);
    size_t sv_bytes = host_sv_vectors.size() * sizeof(custom_data_t);
    size_t coeffs_bytes = host_dual_coeffs.size() * sizeof(custom_data_t);
    size_t predictions_bytes = host_predictions.size() * sizeof(custom_data_t);

    OCL_CHECK(err, cl::Buffer buffer_samples(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
              samples_bytes, host_samples.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_sv(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
              sv_bytes, host_sv_vectors.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_coeffs(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
              coeffs_bytes, host_dual_coeffs.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_predictions(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
              predictions_bytes, host_predictions.data(), &err));

    int narg = 0;
    custom_data_t rho_val = (custom_data_t)model.rho;
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, buffer_samples));
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, buffer_sv));
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, buffer_coeffs));
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, buffer_predictions));
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, dataset.num_samples));
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, model.total_sv));
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, num_features));
    OCL_CHECK(err, err = krnl_svm.setArg(narg++, rho_val));

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_sv, buffer_coeffs}, 0));
        q.finish();

    // --- 6. Pure Hardware Timing Exec ---
    auto t_hw_start = std::chrono::high_resolution_clock::now();

    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_samples, buffer_sv, buffer_coeffs}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_samples}, 0));

    OCL_CHECK(err, err = q.enqueueTask(krnl_svm));

    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_predictions}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());

    auto t_hw_end = std::chrono::high_resolution_clock::now();

    // --- 7. Verification and Profiling ---
    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (int i = 0; i < dataset.num_samples; ++i) {
        int actual = dataset.labels[i];
        int predicted = (host_predictions[i] >= 0.0) ? 1 : -1;

        if (actual == 1 && predicted == 1) TP++;
        else if (actual == -1 && predicted == -1) TN++;
        else if (actual == -1 && predicted == 1) FP++;
        else if (actual == 1 && predicted == -1) FN++;
    }

    double total_hw_time_ms = std::chrono::duration<double, std::milli>(t_hw_end - t_hw_start).count();
    double avg_latency_us = (total_hw_time_ms * 1000.0) / dataset.num_samples;
    double throughput_vps = (dataset.num_samples / (total_hw_time_ms / 1000.0));
    double accuracy = (double)(TP + TN) / dataset.num_samples * 100.0;
    double total_pipeline_time_ms = model_load_time_ms + dataset_load_time_ms + total_hw_time_ms;

    std::cout << "\n========== FPGA Accelerated Metrics ==========" << std::endl;
    std::cout << "Confusion Matrix         : TP=" << TP << ", TN=" << TN << ", FP=" << FP << ", FN=" << FN << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy                 : " << accuracy << " %" << std::endl;
    std::cout << "HW Latency/Sample        : " << avg_latency_us << " us/sample" << std::endl;
    std::cout << "HW Throughput            : " << throughput_vps << " samples/sec" << std::endl;

    std::cout << "\n========== Complete System Timing Profile ==========" << std::endl;
    std::cout << "1. Model Parsing Time (Software)   : " << model_load_time_ms << " ms" << std::endl;
    std::cout << "2. Dataset Parsing Time (Software) : " << dataset_load_time_ms << " ms" << std::endl;
    std::cout << "3. Hardware Execution Time (FPGA)  : " << total_hw_time_ms << " ms" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Total Pipeline Time                : " << total_pipeline_time_ms << " ms" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    return 0;
}
