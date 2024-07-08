#include "model.h"
#include "params.h"
#include "reader.h"
#include <cmath>
#include <thread>

#include "shortcuts.h"


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
    Layer& c_layer = layers [current_layer_count];
    c_layer.neuron_count = neuron_count;
    c_layer.input_count = current_layer_count > 0 ? layers [current_layer_count - 1].neuron_count : INPUT;
    current_layer_count ++;
    
    int b_weights = sizeof (double) * c_layer.input_count * c_layer.neuron_count;
    int b_biases = sizeof (double) * c_layer.neuron_count;
    cudaMalloc (&(c_layer.weights), b_weights);
    cudaMalloc (&(c_layer.biases), b_biases);
    cudaMemcpy (c_layer.weights, weights, b_weights, cudaMemcpyHostToDevice);
    cudaMemcpy (c_layer.biases, biases, b_biases, cudaMemcpyHostToDevice);
}


/**
 * Free memory taken by the general model
 */
void Model::free () {
    for (int i = 0; i < LAYERS; i ++) {
        cudaFree (layers [i].weights);
        cudaFree (layers [i].biases);
    }
    delete[] layers;
}


/** */
__global__ void reset_layer_gpu (int n, double* outputs) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        outputs [id] = 0;
    }
}


/** */
__global__ void process_gpu (
    int input_count, int neuron_count,
    double* inputs, double* weights, double* outputs
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //outputs [u * neuron_count + i] += inputs [u * input_count + j] * weights [i * input_count + j];
    if (id < input_count * neuron_count) {
        atomicAdd (outputs + id % input_count, inputs [id % neuron_count] * weights [id]);
    }
}


/** */
__global__ void add_bias_gpu (int batch_size, int n, double* outputs, double* biases) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicAdd (outputs + id, biases [id % batch_size]);
    }
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

    int block_size = 1024;
    int grid_size = (int) ceil ((double) batch_size * c_layer.neuron_count / block_size);
    int layer_grid_size = (int) ceil ((double) c_layer.input_count * c_layer.neuron_count / block_size);
    reset_layer_gpu <<<grid_size, block_size>>> (batch_size * c_layer.neuron_count, c_data);
    for (int u = 0; u < batch_size; u ++) {
        process_gpu <<<layer_grid_size, block_size>>> (c_layer.input_count, c_layer.neuron_count,
            input + u * c_layer.input_count, c_layer.weights, c_data + u * c_layer.neuron_count);
    }
    add_bias_gpu <<<grid_size, block_size>>> (batch_size, c_layer.neuron_count, c_data, c_layer.biases);
    
    /*// For each neurone in this layer from each batch
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
    }*/
}


/** */
__global__ void relu_gpu (int n, double* outputs) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        outputs [id] *= (outputs [id] > 0);
    }
}


/**
 * Activate the given layer using ReLU
 */
void Model::relu (int layer) {
    double* c_data = data [layer + 1];
    int total = layers [layer].neuron_count * batch_size;
    
    int block_size = 1024;
    int grid_size = (int) ceil ((double) total / block_size);
    relu_gpu <<<grid_size, block_size>>> (total, c_data);
    
    /*for (int i = 0; i < total; i ++) {
        c_data [i] *= (c_data [i] > 0);
    }*/
}


/** */
__global__ void expsum_gpu (int n, double* outputs, double* exp_sum) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicAdd (exp_sum, exp (outputs [id]));
    }
}


/** */
__global__ void softmax_gpu (int n, double* outputs, double* exp_sum) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        outputs [id] = exp (outputs [id]) / *exp_sum;
    }
}


/**
 * Activate the last layer using softmax
 */
void Model::softmax () {
    // Locate the last layer and its outputs
    int categories = layers [LAYERS - 1].neuron_count;
    double* c_data = data [LAYERS];

    double zero = 0;
    double* exp_sum;
    cudaMalloc (&exp_sum, sizeof (double));
    for (int u = 0; u < batch_size; u ++) {
        cudaMemcpy (exp_sum, &zero, sizeof (double), cudaMemcpyHostToDevice);
        expsum_gpu <<<categories, 1>>> (categories, c_data + u * categories, exp_sum);
        softmax_gpu <<<categories, 1>>> (categories, c_data + u * categories, exp_sum);
    }
    cudaFree (exp_sum);
    cudaMemcpy (results, c_data, batch_size * categories * sizeof (double), cudaMemcpyDeviceToHost);

    for (int u = 0; u < batch_size; u ++) {
        double max = 0;
        for (int i = 0; i < categories; i ++) {
            if (results [i] > max) {
                max = results [i];
                outputs [u] = i;
            }
        }
    }

    /*// For each output in last layer from each batch
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
    }*/
}


/**
 * Model constructor
 * Takes a batch size as a parameter
 */
Model::Model (int batch_size) {
    this -> batch_size = batch_size;
    this -> current_input = 0;
    this -> ready = 0;
    input = new double [INPUT * batch_size];
    data = new double* [LAYERS + 1];
    cudaMalloc (&(data [0]), sizeof (double) * INPUT * batch_size);
    for (int i = 1; i <= LAYERS; i ++) {
        cudaMalloc (&(data [i]), sizeof (double) * layers [i - 1].neuron_count * batch_size);
    }
    results = new double [batch_size * L7];
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
    std::thread t (read_input, filename, input + current_input * INPUT, &ready, free_readers);
    t.detach ();
    (*free_readers) --;
    current_input ++;
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

    // Copy inputs to device
    cudaMemcpy (data [0], input, sizeof (double) * INPUT * batch_size, cudaMemcpyHostToDevice);

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
    delete[] input;
    for (int i = 0; i <= LAYERS; i ++) {
        cudaFree (data [i]);
    }
    delete[] data;
    delete[] results;
    delete[] outputs;
    delete[] mappings;
}