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
        void process (double* input, bool is_output);
    };

    // Model's layers and their count
    int layer_count, current_layer_count;
    Layer* layers;

    // Size of an input array
    int input_size;


public:

    // The constuctor
    Model (int layers_count, int input_size);

    // Add a new layer to the model
    void add_layer (int neuron_count, double* weights, double* biases);

    // Forward pass
    int forward_pass (double* input);

    // The destructor
    ~Model ();
};

#endif // MODEL_H