#include "model.h"
#include <cmath>

using namespace std;

// Process current layer in forward propagation
// Takes input to it and a flag whether it's the last layer in a model
void Model::Layer::process (double* input, bool is_output) {
    for (int i = 0; i < neuron_count; i ++) {
        outputs [i] = 0;
        for (int j = 0; j < input_count; j ++) {
            outputs [i] += input [j] * weights [i*input_count + j];
        }
        outputs [i] += biases [i];
    }

    // Activate outputs
    if (is_output) {
        // Softmax
        double exp_sum = 0;
        for (int i = 0; i < neuron_count; i ++) {
            exp_sum += exp (outputs [i]);  // Possible optimisation: use simpler exp
        }
        for (int i = 0; i < neuron_count; i ++) {
            outputs [i] = exp (outputs [i]) / exp_sum;
        }
    }
    else {
        // ReLU
        for (int i = 0; i < neuron_count; i ++) {
            if (outputs [i] < 0) {
                outputs [i] = 0;
            }
        }
    }
}


// The constuctor
// Takes an input size as a parameter
Model::Model (int layers_count, int input_size) {
    this -> layer_count = layers_count;
    this -> current_layer_count = 0;
    this -> layers = new Layer [layers_count];
    this -> input_size = input_size;
}


// Add a new layer to the model
// Takes number of neurons in this layers along with their weights and biases
void Model::add_layer (int neuron_count, double* weights, double* biases) {
    layers [current_layer_count] = {
        .neuron_count = neuron_count,
        .input_count = current_layer_count > 0 ?
            layers [current_layer_count - 1].neuron_count : input_size,
        .weights = weights,
        .biases = biases,
        .outputs = new double [neuron_count]
    };
    current_layer_count ++;
}


// Forward pass
// Takes input array as a parameter and returns an index of a most similar letter
int Model::forward_pass (double* input) {
    // Process input -> first layer
    layers [0].process (input, false);

    // Process layer K -> layer K + 1
    for (int i = 1; i < layer_count - 1; i ++) {
        layers [i].process (layers [i - 1].outputs, false);
    }

    // Process pre-last layer -> last layer
    Layer& last_layer = layers [layer_count - 1];
    last_layer.process (layers [layer_count - 2].outputs, true);

    // Find maximum output and its position
    int pos = 0;
    double max = 0;
    for (int i = 0; i < last_layer.neuron_count; i ++) {
        if (last_layer.outputs [i] > max) {
            max = last_layer.outputs [i];
            pos = i;
        }
    }
    return pos;
}


// The destructor
// Frees used memory, namely all neurons' weights and biases
Model::~Model () {
    for (int i = 0; i < layer_count; i ++) {
        delete[] layers [i].weights;
        delete[] layers [i].biases;
        delete[] layers [i].outputs;
    }
}