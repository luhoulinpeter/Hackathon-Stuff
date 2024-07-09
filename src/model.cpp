#include "model.h"
#include "params.h"
#include "reader.h"
#include <cmath>
#include <thread>
#include "helpers.h"


/**
 * Declare model static variables
 */
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
    if (current_layer_count == LAYERS) { return; }
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
 * Faster exponential function
 * Works twice as fast as a regular one, but produces error the greater the number is
 */
inline double fexp (double a) noexcept (true) {
    union { double d; long long x; } u;
    u.x = (long long) (6497320848556798LL * a + 4606794787981043020);
    return u.d;
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
            exp_sum += fexp (c_data [u * categories + i]);
        }

        // Calculate output
        double max = 0;
        for (int i = 0; i < categories; i ++) {
            double& c_out = c_data [u * categories + i];
            c_out = fexp (c_out) / exp_sum;
            if (c_out > max) {
                max = c_out;
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
    this -> current_input = 0;
    this -> ready = 0;
    data = new double* [LAYERS + 1];
    data [0] = new double [INPUT * batch_size];
    for (int i = 1; i <= LAYERS; i ++) {
        data [i] = new double [layers [i - 1].neuron_count * batch_size];
    }
    outputs = new int [batch_size];
    mappings = new int [batch_size];
}


/**
 * Return true if all inputs are covered, otherwise false
 */ 
bool Model::is_ready () {
    return current_input == batch_size;
}


/**
 * Read tensor into current input array
 * Takes tensor filename, a position for mapping, and a pointer to the number of free readers
 */
void Model::process_input (const std::string& filename, int pos, std::atomic_int* free_readers) {
    mappings [current_input] = pos;
    std::thread t (read_input, filename, data [0] + current_input * INPUT, &ready, free_readers);
    t.detach ();
    current_input ++;
    (*free_readers) --;
}


/**
 * Forward pass
 * Takes an auxiliary array to store results in,
 * a queue of models to make itself available again,
 * and a sub-batch parameter (optional)
*/
void Model::forward_pass (char* aux, tq* models, int sub_batch) {
    // Save original batch size
    int original_batch = batch_size;
    if (sub_batch > 0 && sub_batch < batch_size) {
        batch_size = sub_batch;
    }

    // Wait for all inputs to be read
    while (ready != batch_size) {IDLE}

    // Activate layer K-1, then process it to layer K
    for (int i = 0; i < LAYERS - 1; i ++) {
        process (i);
        relu (i);
    }

    // Process and activate the last layer and get outputs
    process (LAYERS - 1);
    softmax ();

    // Write results to the auxilary output array
    for (int i = 0; i < batch_size; i ++) {
        int res = outputs [i];
        aux [mappings [i]] = res % 2 ? char (97 + res / 2) : char (65 + res / 2);
    }

    // Restore original batch size, current input and ready count
    batch_size = original_batch;
    current_input = 0;
    ready = 0;

    // Make this model available again by adding it to the queue of available models
    models -> push (this);
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
    delete[] mappings;
}