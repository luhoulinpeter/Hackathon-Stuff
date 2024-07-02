#include "model.h"
#include "params.h"
#include <cmath>

Model::Layer* Model::layers;
int Model::current_layer_count;


/**
 * Initialize the general model
 */
void Model::init () {
    layers = new Layer [LAYERS];
    current_layer_count = 0;
}


/**
 * Add a new layer to the model
 * Takes number of neurons in this layers along with their weights and biases
 */
void Model::add_layer (int neuron_count, double* weights, double* biases) {
    layers [current_layer_count] = {
        neuron_count,
        current_layer_count > 0 ? layers [current_layer_count - 1].neuron_count : INPUT,
        weights,
        biases
    };
    current_layer_count ++;
}


/**
 * Free memory taken by the general model
 */
void Model::free () {
    for (int i = 0; i < LAYERS; i ++) {
        delete[] layers [i].weights;
        delete[] layers [i].biases;
    }
    delete[] layers;
}


/**
 * Layer processing
 * Processes the given layer by taking an input to it
*/
void Model::process (int layer) {
    // Locate current layer, its inputs and outputs
    Layer& c_layer = layers [layer];
    double* input = data [layer];
    double* c_data = data [layer + 1];
    
    // For each neurone in this layer from each batch
    for (int u = 0; u < batch_size; u ++) {
        for (int i = 0; i < c_layer.neuron_count; i ++) {
            double& c_out = c_data [u * c_layer.neuron_count + i];

            // Compute the output for current neurone
            c_out = 0;
            for (int j = 0; j < c_layer.input_count; j ++) {
                c_out += input [u * c_layer.input_count + j] * c_layer.weights [i * c_layer.input_count + j];
            }
            c_out += c_layer.biases [i];
        }
    }
}


/**
 * Activate the given layer using ReLU
 */
void Model::relu (int layer) {
    double* c_data = data [layer + 1];
    int total = layers [layer].neuron_count * batch_size;
    for (int i = 0; i < total; i ++) {
        c_data [i] *= (c_data [i] > 0);
    }
}


/**
 * Activate the last layer using softmax
 */
void Model::softmax () {
    // Locate the last layer and its outputs
    int categories = layers [LAYERS - 1].neuron_count;
    double* c_data = data [LAYERS];

    // For each output in last layer from each batch
    for (int u = 0; u < batch_size; u ++) {
        double exp_sum = 0;

        // Calculate exp sum
        for (int i = 0; i < categories; i ++) {
            exp_sum += exp (c_data [u * categories + i]);
        }

        // Calculate output
        double max = 0;
        for (int i = 0; i < categories; i ++) {
            c_data [i] = exp (c_data [u * categories + i]) / exp_sum;
            if (c_data [i] > max) {
                max = c_data [i];
                outputs [u] = i;
            }
        }
    }
}


/**
 * Model constructor
 * Takes a batch size as a parameter
 */
Model::Model (int batch_size) {
    this -> batch_size = batch_size;
    data = new double* [LAYERS + 1];
    data [0] = new double [INPUT * batch_size];
    for (int i = 1; i <= LAYERS; i ++) {
        data [i] = new double [layers [i - 1].neuron_count * batch_size];
    }
    outputs = new int [batch_size];
}


/**
 * Get inputs
 */
double* Model::get_inputs () {
    return data [0];
}


/**
 * Forward pass with limited batch size
 */
int* Model::sub_forward_pass (int sub_batch) {
    int original_batch = batch_size;
    if (sub_batch > 0 && sub_batch < batch_size) {
        batch_size = sub_batch;
    }

    forward_pass ();
    
    batch_size = original_batch;
    return outputs;
}


/**
 * Forward pass
 * Takes input array as a parameter and returns an array of indexes of the most similar letters
*/
int* Model::forward_pass () {
    // Activate layer K-1, then process it to layer K
    for (int i = 0; i < LAYERS - 1; i ++) {
        process (i);
        relu (i);
    }

    // Process and activate the last layer and get outputs
    process (LAYERS - 1);
    softmax ();

    return outputs;
}


/**
 * Model destructor
 * Frees all model outputs and data
 */
Model::~Model () {
    delete[] outputs;
    for (int i = 0; i <= LAYERS; i ++) {
        delete[] data [i];
    }
    delete[] data;
}