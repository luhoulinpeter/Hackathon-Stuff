#ifndef MODEL_H
#define MODEL_H

#include "eigenreader.h"

MatrixXd pass_through_model(MatrixXd input, std::vector<MatrixXd> Weights, std::vector<MatrixXd> Biases);

void relu(MatrixXd *layer_output);

void softmax(MatrixXd *layer_output);

struct output_batch {
    int input_numbers[BATCH_SIZE] = {0};
    char character[BATCH_SIZE] = {0};
};

struct output_batch create_output_map(int *input_map, MatrixXd *outputs);

#endif // MODEL_H