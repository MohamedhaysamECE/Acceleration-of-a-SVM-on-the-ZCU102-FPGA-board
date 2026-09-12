#include "libsvm_parser.h"

SVMModel parse_libsvm_model(const std::string& model_path, int num_features) {
    SVMModel model;
    model.num_features = num_features;
    model.gamma = 0.0;
    model.coef0 = 0.0;
    model.degree = 3;
    model.rho = 0.0;
    model.total_sv = 0;

    std::ifstream file(model_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open model file: " << model_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    bool sv_header_reached = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line == "SV") {
            sv_header_reached = true;
            break;
        }

        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "kernel_type") iss >> model.kernel_type;
        else if (key == "gamma") iss >> model.gamma;
        else if (key == "coef0") iss >> model.coef0;
        else if (key == "degree") iss >> model.degree;
        else if (key == "rho") iss >> model.rho;
        else if (key == "total_sv") iss >> model.total_sv;
    }

    if (!sv_header_reached) {
        std::cerr << "Error: Invalid LIBSVM model format. Missing 'SV' label." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    model.support_vectors.resize(model.total_sv, std::vector<double>(num_features, 0.0));
    model.dual_coeffs.resize(model.total_sv, 0.0);

    for (int i = 0; i < model.total_sv; ++i) {
        if (!std::getline(file, line)) break;
        std::istringstream iss(line);
        
        iss >> model.dual_coeffs[i];
        
        std::string feature_pair;
        while (iss >> feature_pair) {
            size_t colon_pos = feature_pair.find(':');
            if (colon_pos != std::string::npos) {
                int index = std::stoi(feature_pair.substr(0, colon_pos)) - 1; 
                double val = std::stod(feature_pair.substr(colon_pos + 1));
                if (index >= 0 && index < num_features) {
                    model.support_vectors[i][index] = val;
                }
            }
        }
    }

    file.close();
    return model;
}

Dataset parse_libsvm_dataset(const std::string& dataset_path, int num_features) {
    Dataset ds;
    ds.num_features = num_features;
    ds.num_samples = 0;

    std::ifstream file(dataset_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open dataset file: " << dataset_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (ds.num_samples >= 500) {
                    break;
                }
        std::istringstream iss(line);
        
        int label;
        iss >> label;
        ds.labels.push_back(label);

        std::vector<double> feat(num_features, 0.0);
        std::string feature_pair;
        while (iss >> feature_pair) {
            size_t colon_pos = feature_pair.find(':');
            if (colon_pos != std::string::npos) {
                int index = std::stoi(feature_pair.substr(0, colon_pos)) - 1;
                double val = std::stod(feature_pair.substr(colon_pos + 1));
                if (index >= 0 && index < num_features) {
                    feat[index] = val;
                }
            }
        }
        ds.features.push_back(feat);
        ds.num_samples++;
    }

    file.close();
    return ds;
}
