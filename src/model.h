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
        double* outputs;

        // Process current layer in forward propagation
        // Takes input to it and a flag whether it's the last layer in a model
        void process (double* input, bool is_output);
    };

    // Model's layers and their count
    int layer_count, current_layer_count;
    Layer* layers;

    // Size of an input array
    int input_size;


public:

    // The constuctor
    // Takes an input size as a parameter
    Model (int layers_count, int input_size);

    // Add a new layer to the model
    // Takes number of neurons in this layers along with their weights and biases
    void add_layer (int neuron_count, double* weights, double* biases);

    // Forward pass
    // Takes input array as a parameter and returns an index of a most similar letter
    int forward_pass (double* input);

    // The destructor
    // Frees used memory, namely all neurons' weights and biases
    ~Model ();
};

#endif // MODEL_H