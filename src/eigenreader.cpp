#include "eigenreader.h"
#include <fstream>
#include <sstream>
#include <iostream>


void parse_line (const std::string& line, MatrixXd *values) {
    int count = 0;
    std::istringstream stream(line);
    std::string token;

    while (getline(stream, token, ',')) {
        (*values)(count++) = stod(token);
    }
}

std::vector<MatrixXd> init_weights() {
    std::vector<MatrixXd> weights = {
        {L1, INPUT},
        {L2, L1},
        {L3, L2},
        {L4, L3},
        {L5, L4},
        {L6, L5},
        {L7, L6}
    };
    return weights;
}

std::vector<MatrixXd> init_biases() {
    std::vector<MatrixXd> biases = {
        {L1, 1},
        {L2, 1},
        {L3, 1},
        {L4, 1},
        {L5, 1},
        {L6, 1},
        {L7, 1}
    };
    return biases;
}

void read_weights_biases(const std::string& filename, std::vector<MatrixXd> *weights, std::vector<MatrixXd> *biases) {
    std::ifstream file (filename);
    std::string line;

    for (int i=0;i<LAYERS; i++) {
        getline(file, line);
        getline(file, line);
        parse_line (line, &(*weights)[i]);
        getline(file, line);
        getline(file, line);
        parse_line (line, &(*biases)[i]);
    }
}

// Function to collect all file paths in the directory
std::vector<std::string> collect_file_paths(const std::string& directory) {
    std::vector<std::string> file_paths;
    
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        file_paths.push_back(entry.path().string());
    }
    return file_paths;
}

/**
 * Read values from file to an array
 * Takes a filename to read an input from and returns an array of values
 */
std::vector<input_batch> read_inputs(const std::string& dir_path) {
    // list of file_paths
    std::vector<std::string> file_paths = collect_file_paths(dir_path);
    //std::cout << file_paths[0] << "\n\n";

    //initialsing vector of input_batch structs
    int num_batches = file_paths.size() / BATCH_SIZE;
    std::vector<input_batch> batches(num_batches);

    //writing to batches
    std::string path;
    int index;
    //int count = 0;
    for (int i=0; i<num_batches; i++) {
        for (int j=0; j<BATCH_SIZE; j++) {
            path = file_paths[j+(i*BATCH_SIZE)];
            //getting input line from filepath
            std::ifstream file(path);
            std::string line;
            getline(file, line);
            file.close();
            
            // write input into matrix
            write_input(line, &(batches[i].inputs), j);

            // write input number (mapping)
            index = std::stoi(path.substr(dir_path.size() + 1, path.size() - 8 - dir_path.size()));
            //std::cout << index << "count" << count++ << "\n";
            batches[i].input_number[j] = index;
        }
    }

    // printing batches to test
    // std::cout << "\n" << num_batches << "\n" << batches[0].input_number[0] << "\n";
    // for (int i=0; i<num_batches; i++) {
    //     std::cout << batches[i].inputs << "\n\n\n";
    // }

    // printing batch input numbers
    // for (int i=0; i<num_batches; i++) {
    //     for (int j=0; j<BATCH_SIZE; j++) {
    //         std::cout << batches[i].input_number[j] << ", ";
    //     }
    //     std::cout << "\n";
    // }

    return batches;
}

// bascially modified parse_line that allows writing of individual columns of a matrix
void write_input(const std::string& line, MatrixXd *values, int col) {
    int count = 0;
    std::istringstream stream(line);
    std::string token;

    while (getline(stream, token, ',')) {
        (*values)(count++, col) = stod(token);
    }
}