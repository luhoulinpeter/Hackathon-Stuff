#ifndef MODEL_H
#define MODEL_H

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
    double** data;
    int* outputs;

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

    // Get inputs
    double* get_inputs ();

    // Forward pass with limited batch size
    int* sub_forward_pass (int sub_batch);

    // Forward pass
    int* forward_pass ();

    // The destructor
    ~Model ();
};

#endif // MODEL_H