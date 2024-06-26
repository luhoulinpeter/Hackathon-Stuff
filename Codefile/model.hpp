#include <list>

using namespace std;

// Neural network model class
class Model {
private:
    // Size of input array
    int input_size;

    // Layer structure
    struct Layer {
        int neurons_count, prev_layer_count;
        double* weights;
        double* biases;
    };

    // Model's layers
    list <Layer> layers;

public:
    // The constuctor
    // Takes an input size as a parameter
    Model (int input_size) {
        this -> input_size = input_size;
    }

    // Add a new layer to the model
    // Takes number of neurons in this layers along with their weights and biases
    void add_layer (int neurons_count, double* weights, double* biases) {
        layers.push_back ({
            .neurons_count = neurons_count,
            .prev_layer_count = layers.empty () ? input_size : layers.back ().neurons_count,
            .weights = weights,
            .biases = biases
        });
    }

    // Forward pass
    // Takes input array as a parameter and returns an index of a most similar letter
    int forward_pass (double* input) {
        //TODO
    }

    // The destructor
    // Frees used memory, namely all neurons' weights and biases
    ~Model () {
        for (Layer& layer : layers) {
            delete layer.weights;
            delete layer.biases;
        }
    }
};