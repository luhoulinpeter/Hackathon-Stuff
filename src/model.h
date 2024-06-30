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

    // Process the given layer in forward propagation
    void process (int layer, double* input);

    // Activate the given layer output using ReLU
    void relu (int layer);

    // Activate the last layer output using
    void softmax ();

    // Return an output index array
    int* select ();


public:

    // Model initialization
    static void init ();

    // Add a new layer to the model
    static void add_layer (int neuron_count, double* weights, double* biases);

    // Model uninitialization
    static void free ();

    // The constuctor
    Model (int batch_size);

    // Forward pass
    int* forward_pass (double* input);

    // The destructor
    ~Model ();
};

#endif // MODEL_H