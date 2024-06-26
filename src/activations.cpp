/*
    Implements softmax and rectified linear activation functions
*/

#include "activations.h"
#include <cmath>

vector<long double> softmax(vector<long double> input, int length) {
    vector<long double> output;
    long double exp_sum = 0;

    // sum exponentials
    for (int i=0; i<length; i++) {
        exp_sum += exp(input[i]);
    }

    // calculating output softmax array
    for (int i=0; i<length; i++) {
        output[i] = input[i] / exp_sum;
    }

    return output;
}
