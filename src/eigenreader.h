#ifndef READER_H
#define READER_H

#include <string>
#include <filesystem>
#include "params.h"
#include "Dense"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;

// input batch struct
struct input_batch {
    int input_number[BATCH_SIZE] = {0};
    MatrixXd inputs;
    input_batch() : inputs(INPUT, BATCH_SIZE) {} //initialiser list, constructor
};

// initialises all weights matrices as vector of matrices
std::vector<MatrixXd> init_weights();

// initialises all bias vectors as vector of vectors
std::vector<MatrixXd> init_biases();

// // Reads weights and biases from file to parameters
void read_weights_biases(const std::string& filename, std::vector<MatrixXd> *weights, std::vector<MatrixXd> *biases);


// Takes a line to be parse and array to insert parsed values into
void parse_line (const std::string& line, MatrixXd *values);

// Function to collect all file paths in the directory
std::vector<std::string> collect_file_paths(const std::string& directory);

// Read values from file to vector of input_batch struct
std::vector<input_batch> read_inputs(const std::string& dir_path);

// bascially modified parse_line that allows writing of individual columns of a matrix
void write_input(const std::string& line, MatrixXd *values, int col);

#endif 