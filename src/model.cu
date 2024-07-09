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
    if (current_layer_count == LAYERS) {
        cudaDeviceSynchronize ();
        return;
    }
    Layer& c_layer = layers [current_layer_count];
    
    // Initialize layer
    c_layer.neuron_count = neuron_count;
    c_layer.input_count = current_layer_count > 0 ? layers [current_layer_count - 1].neuron_count : INPUT;
    current_layer_count ++;
    
    // Allocate space and copy data to device
    int b_weights = sizeof (double) * c_layer.input_count * c_layer.neuron_count;
    int b_biases = sizeof (double) * c_layer.neuron_count;
    cudaMalloc (&(c_layer.weights), b_weights);
    cudaMemcpyAsync (c_layer.weights, weights, b_weights, cudaMemcpyHostToDevice);
    cudaMalloc (&(c_layer.biases), b_biases);
    cudaMemcpyAsync (c_layer.biases, biases, b_biases, cudaMemcpyHostToDevice);

    // Free host weights and biases
    delete[] weights;
    delete[] biases;
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


/**
 * Cuda kernel code to set values of a given array to 0
 */
__global__ void clear_gpu (int n, double* data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        data [id] = 0;
    }
}


/**
 * Cuda kernel code to process a layer (perform matrix multiplication)
 */
__global__ void process_gpu (
    int batch_size, int input_count, int neuron_count,
    double* inputs, double* weights, double* outputs
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < batch_size * neuron_count) {
        for (int j = 0; j < input_count; j ++) {
            atomicAdd (
                outputs + id,
                inputs [id / neuron_count * input_count + j] *
                weights [id % neuron_count * input_count + j]
            );
        }
    }
}


/**
 * Cuda kernel code to add biases to outputs
 */
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

    // Setup the block and grid sizes
    int block_size = 1024;
    int grid_size = (int) ceil ((double) batch_size * c_layer.neuron_count / block_size);

    // Reset all outputs
    clear_gpu <<<grid_size, block_size>>> (batch_size * c_layer.neuron_count, c_data);

    // Perform matrix multiplication
    process_gpu <<<grid_size, block_size>>> (
        batch_size, c_layer.input_count, c_layer.neuron_count, input, c_layer.weights, c_data
    );

    // Add bias to the outputs
    add_bias_gpu <<<grid_size, block_size>>> (batch_size, c_layer.neuron_count, c_data, c_layer.biases);
}


/**
 * Cuda kernel code to calculate ReLU for given array of outputs
 */
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
    
    // Setup the block and grid sizes
    int block_size = 1024;
    int grid_size = (int) ceil ((double) total / block_size);

    // Perform ReLU activation
    relu_gpu <<<grid_size, block_size>>> (total, c_data);
}


/**
 * Cuda kernel code to calculate exponential sums for a batch of given outputs
 */
__global__ void expsum_gpu (int n, int categories, double* outputs, double* exp_sums) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        atomicAdd (exp_sums + id / categories, exp (outputs [id]));
    }
}


/**
 * Cuda kernel code to activate the given batch of outputs with exponential sums provided
 */
__global__ void softmax_gpu (int n, int categories, double* outputs, double* exp_sums) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        outputs [id] = exp (outputs [id]) / exp_sums [id / categories];
    }
}


/**
 * Cuda kernel code to select outputs with highest probabilities
 */
__global__ void select_gpu (int batch_size, int categories, double* outputs, int* results) {
    double max;
    for (int u = 0; u < batch_size; u ++) {
        max = 0;
        for (int i = 0; i < categories; i ++) {
            if (outputs [u * categories + i] > max) {
                max = outputs [u * categories + i];
                results [u] = i;
            }
        }
    }
}


/**
 * Activate the last layer using softmax
 */
void Model::softmax () {
    // Locate the last layer and its outputs
    int categories = layers [LAYERS - 1].neuron_count;
    int total = batch_size * categories;
    double* c_data = data [LAYERS];

    // Reset all exponential sums
    clear_gpu <<<batch_size, 1>>> (batch_size, expsums);

    // Setup the block and grid sizes
    int block_size = 1024;
    int grid_size = (int) ceil ((double) total / block_size);

    // Calculate exponential sums
    expsum_gpu <<<grid_size, block_size>>> (total, categories, c_data, expsums);

    // Perform softmax activation
    softmax_gpu <<<grid_size, block_size>>> (total, categories, c_data, expsums);

    // Process results
    select_gpu <<<1, 1>>> (batch_size, categories, c_data, results);
    
    // Copy results back to host
    cudaMemcpy (outputs, results, batch_size * sizeof (int), cudaMemcpyDeviceToHost);
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
    cudaMalloc (&expsums, sizeof (double) * batch_size);
    cudaMalloc (&results, sizeof (int) * batch_size);
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
    cudaFree (expsums);
    cudaFree (results);
    for (int i = 0; i <= LAYERS; i ++) {
        cudaFree (data [i]);
    }
    delete[] data;
    delete[] outputs;
    delete[] mappings;
}