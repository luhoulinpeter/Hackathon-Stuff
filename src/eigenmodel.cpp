#include "eigenmodel.h"
#include <cmath>
#include <iostream>

// takes a singel input matrix, thus some external function must pass in individual matrices
MatrixXd pass_through_model(MatrixXd input, std::vector<MatrixXd> Weights, std::vector<MatrixXd> Biases) {
    //relu stuff
    MatrixXd temp = input;
    for (int i=0; i<(LAYERS-1); i++) {
        //std::cout << i << "\n";
        // multiply weights
        temp = Weights[i]*temp;
        // add biases
        for (int j=0; j<BATCH_SIZE; j++) {
            temp.col(j) += Biases[i];
        }
        // passing through relu
        relu(&temp);
    }

    // processing last layer with softmax
    temp = Weights[LAYERS-1]*temp;
    for (int j=0; j<BATCH_SIZE; j++) {
        temp.col(j) += Biases[LAYERS-1];
    }
    softmax(&temp);

    return temp;
}

void relu(MatrixXd *layer_output) {
    for (int i=0; i<(*layer_output).size(); i++) {
        (*layer_output)(i) = (*layer_output)(i)*((*layer_output)(i) > 0);
    }
}

void softmax(MatrixXd *layer_output) {
    for (int i=0; i<BATCH_SIZE; i++) {
        double exp_sum = 0;
        // calculating sum
        for (int j=0; j<L7; j++) {
            exp_sum += exp((*layer_output)(j, i));
        }
        //std::cout << exp_sum << "expsum\n\n";
        // writing to output vector
        for (int j=0; j<L7; j++) {
            (*layer_output)(j, i) = exp((*layer_output)(j, i)) / exp_sum;
        }
    }
    
}

struct output_batch create_output_map(int *input_map, MatrixXd *outputs) {
    struct output_batch outbatch;
    // finding index corresponding to max values for all batches
    for (int i=0; i<BATCH_SIZE; i++) {
        // copy input map over
        outbatch.input_numbers[i] = input_map[i];
        // finding max index
        double max = 0;
        int max_index = 0;
        for (int j=0; j<L7; j++) {
            if ((*outputs)(j, i) > max) {
                max_index = j;
                max = (*outputs)(j, i);
            }
        }
        // mapping max index to the ascii character, then saving to output batch
        outbatch.character[i] = max_index%2 ? char(97 + max_index / 2) : char(65 + max_index / 2);
    }
    return outbatch;
}