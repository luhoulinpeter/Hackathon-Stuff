#include "model.h"
#include <cmath>

/**
 * Layer processing
 * Processes current layer by taking an input to it
*/
void Model::Layer::process (double* input) {
    for (int i = 0; i < neuron_count; i ++) {
        outputs [i] = 0;
        for (int j = 0; j < input_count; j ++) {
            outputs [i] += input [j] * weights [i*input_count + j];
        }
        outputs [i] += biases [i];
    }
}


/**
 * Activates current layer using ReLU
 */
void Model::Layer::relu () {
    for (int i = 0; i < neuron_count; i ++) {
        outputs [i] *= (outputs[i] > 0);
    }
}


/**
 * Activates current layer using softmax
 */
void Model::Layer::softmax () {
    double exp_sum = 0;
    for (int i = 0; i < neuron_count; i ++) {
        exp_sum += exp (outputs [i]);
    }
    for (int i = 0; i < neuron_count; i ++) {
        outputs [i] = exp (outputs [i]) / exp_sum;
    }
}


/**
 * Model constructor
 * Takes a number of layers and a size of an input array as parameters
 */
Model::Model (int layers_count, int input_size) {
    this -> layer_count = layers_count;
    this -> current_layer_count = 0;
    this -> layers = new Layer [layers_count];
    this -> input_size = input_size;
}


/**
 * Add a new layer to the model
 * Takes number of neurons in this layers along with their weights and biases
 */
void Model::add_layer (int neuron_count, double* weights, double* biases) {
    layers [current_layer_count] = {
        neuron_count,
        current_layer_count > 0 ? layers [current_layer_count - 1].neuron_count : input_size,
        weights,
        biases,
        new double [neuron_count]
    };
    current_layer_count ++;
}


/**
 * Forward pass
 * Takes input array as a parameter and returns an index of a most similar letter
*/
int Model::forward_pass (double* input) {
    // Process input to first layer
    layers [0].process (input);
    delete[] input;

    // Activate layer K-1, then process it to layer K
    for (int i = 1; i < layer_count; i ++) {
        // layers [i - 1].activate (false);
        layers [i - 1].relu ();
        layers [i].process (layers [i - 1].outputs);
    }

    // Actvate last layer
    Layer& last_layer = layers [layer_count - 1];
    // last_layer.activate (true);
    last_layer.softmax ();

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


/**
 * Model destructor
 * Frees used memory, namely all layers and their neurons' weights and biases
 */
Model::~Model () {
    for (int i = 0; i < layer_count; i ++) {
        delete[] layers [i].weights;
        delete[] layers [i].biases;
        delete[] layers [i].outputs;
    }
    delete[] layers;
}



/**
 * Activates current layer
 * Takes a flag whether it's the last layer in a model
 */
/*  void Model::Layer::activate (bool is_output) {
    if (is_output) {
        // Softmax
        double exp_sum = 0;
        for (int i = 0; i < neuron_count; i ++) {
            exp_sum += exp (outputs [i]);
        }
        for (int i = 0; i < neuron_count; i ++) {
            outputs [i] = exp (outputs [i]) / exp_sum;
        }
    }
    else {
        // ReLU
        for (int i = 0; i < neuron_count; i ++) {
            outputs [i] *= (outputs[i] > 0);
        }
    }
}   */