#ifndef LIBSVM_PARSER_H
#define LIBSVM_PARSER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdlib>

struct SVMModel {
    std::string kernel_type;
    int total_sv;
    int num_features;
    double gamma;
    double coef0;
    int degree;
    double rho; 
    std::vector<double> dual_coeffs;
    std::vector<std::vector<double>> support_vectors;
};

struct Dataset {
    int num_samples;
    int num_features;
    std::vector<int> labels;
    std::vector<std::vector<double>> features;
};

SVMModel parse_libsvm_model(const std::string& model_path, int num_features);
Dataset parse_libsvm_dataset(const std::string& dataset_path, int num_features);

#endif