#ifndef MODEL_H
#define MODEL_H

#include <atomic>
#include <string>
#include "tq.h"

/**
 * Neural network model class
*/
class Model {
private:

    // Layer structure
    struct Layer {
        int neuron_count, input_count;
        double* weights;
        double* biases;
    };

    // Model's layers and their count
    static Layer* layers;
    static int current_layer_count;

    // Model data
    int batch_size;
    int current_input;
    std::atomic_int ready;
    double* input;
    double** data;
    double* expsums;
    int* results;
    int* outputs;
    int* mappings;

    // Process the given layer in forward propagation
    void process (int layer);

    // Activate the given layer output using ReLU
    void relu (int layer);

    // Activate the last layer output using
    void softmax ();


public:
    // Model initialization
    static void init ();

    // Add a new layer to the model
    static void add_layer (int neuron_count, double* weights, double* biases);

    // Model uninitialization
    static void free ();

    // The constuctor
    Model (int batch_size);

    // Is model has all inputs filled in
    bool is_ready ();

    // Read tensor into input
    void process_input (const std::string& filename, int pos, std::atomic_int* free_readers);

    // Forward pass
    void forward_pass (char* aux, tq* models, int sub_batch);

    // The destructor
    ~Model ();
};

#endif // MODEL_H