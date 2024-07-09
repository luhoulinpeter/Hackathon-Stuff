#include "model.h"
#include "params.h"
#include <cmath>
#include <cstdint>

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
void Model::process (int layer, double* input) {
    // Locate current layer and its outputs
    Layer& c_layer = layers [layer];
    double* c_data = data [layer];
    
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
    int total = layers [layer].neuron_count * batch_size;
    for (int i = 0; i < total; i ++) {
        data [layer] [i] *= (data [layer] [i] > 0);
    }
}


/**
 * exp() using bitshift
 */
inline float fast_exp(float x) {
    union {
        float f;
        int32_t i;
    } val = { x };

    val.i = static_cast<int32_t>(12102203 * x + 1064866805);
    return val.f;
}

/**
 * Activate the last layer using softmax
 */
void Model::softmax () {
    // Locate the last layer and its outputs
    Layer& c_layer = layers [LAYERS - 1];
    double* c_data = data [LAYERS - 1];

    // For each output in last layer from each batch
    for (int u = 0; u < batch_size; u ++) {
        double exp_sum = 0;
        int offset = u * c_layer.neuron_count;

        // Calculate exp sum
        for (int i = 0; i < c_layer.neuron_count; i ++) {
            c_data [offset + i] = exp (c_data [offset + i]);
            exp_sum += c_data [offset + i];
        }

        // Divide exp of each output by the sum
        for (int i = 0; i < c_layer.neuron_count; i ++) {
            c_data [i] = c_data [offset + i] / exp_sum;
        }
    }

}


/**
 * Return an output arrray of indexes
 */
int* Model::select () {
    // Locate the last layer and its outputs
    Layer& c_layer = layers [LAYERS - 1];
    double* c_data = data [LAYERS - 1];

    // For each output in last layer from each batch
    int* outputs = new int [batch_size];
    for (int u = 0; u < batch_size; u ++) {
        double max = 0;

        // Find the largest value and its index
        for (int i = 0; i < c_layer.neuron_count; i ++) {
            if (c_data [i] > max) {
                max = c_data [i];
                outputs [u] = i;
            }
        }
    }
    return outputs;
}


/**
 * Model constructor
 * Takes a batch size as a parameter
 */
Model::Model (int batch_size) {
    this -> batch_size = batch_size;
    data = new double* [LAYERS];
    for (int i = 0; i < LAYERS; i ++) {
        data [i] = new double [layers [i].neuron_count * batch_size];
    }
}


/**
 * Forward pass
 * Takes input array as a parameter and returns an array of indexes of the most similar letters
*/
int* Model::forward_pass (double* input) {
    // Process input to first layer
    process (0, input);
    delete[] input;

    // Activate layer K-1, then process it to layer K
    for (int i = 1; i < LAYERS; i ++) {
        relu (i - 1);
        process (i, data [i - 1]);
    }

    // Actvate last layer
    softmax ();

    // Return outputs
    return select ();
}


/**
 * Model destructor
 * Frees all model outputs and data
 */
Model::~Model () {
    for (int i = 0; i < LAYERS; i ++) {
        delete[] data [i];
    }
    delete[] data;
}