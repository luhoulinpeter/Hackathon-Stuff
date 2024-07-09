#include "reader.h"
#include "params.h"
#include "model.h"
#include <fstream>
#include <sstream>

using namespace std;


/**
 * Parse a line into an array
 * Takes a line to be parse and array to insert parsed values into
 */
void parse_line (const string& line, double* values) {
    int count = 0;
    istringstream stream (line);
    string token;

    while (getline (stream, token, ',')) {
        values [count ++] = stod (token);
    }
}


/**
 * Read values from file to an array
 * Takes a tensor filename, an input array,
 * a pointer to "ready" variable in related model and to a number of free readers
 */
 void read_input (const string& filename, double* input, atomic_int* ready, atomic_int* free_readers) {
    ifstream file (filename);
    string line;
    getline (file, line);
    file.close ();
    parse_line (line, input);

    (*ready) ++;
    (*free_readers) ++;
}


/**
 * Model initialisation
*/
void init_model (const string& wab) {
    // Initialize weigts and biases
    double* weightsL1 = new double [INPUT*L1];
    double* weightsL2 = new double [L1*L2];
    double* weightsL3 = new double [L2*L3];
    double* weightsL4 = new double [L3*L4];
    double* weightsL5 = new double [L4*L5];
    double* weightsL6 = new double [L5*L6];
    double* weightsL7 = new double [L6*L7];

    double* biasesL1 = new double [L1];
    double* biasesL2 = new double [L2];
    double* biasesL3 = new double [L3];
    double* biasesL4 = new double [L4];
    double* biasesL5 = new double [L5];
    double* biasesL6 = new double [L6];
    double* biasesL7 = new double [L7];
    
    // Read weights and biases
    ifstream file (wab);
    string line;
    
    getline (file, line);
    getline (file, line);
    parse_line (line, weightsL1);
    getline (file, line);
    getline (file, line);
    parse_line (line, biasesL1);
    getline (file, line);
    getline (file, line);
    parse_line (line, weightsL2);
    getline (file, line);
    getline (file, line);
    parse_line (line, biasesL2);
    getline (file, line);
    getline (file, line);
    parse_line (line, weightsL3);
    getline (file, line);
    getline (file, line);
    parse_line (line, biasesL3);
    getline (file, line);
    getline (file, line);
    parse_line (line, weightsL4);
    getline (file, line);
    getline (file, line);
    parse_line (line, biasesL4);
    getline (file, line);
    getline (file, line);
    parse_line (line, weightsL5);
    getline (file, line);
    getline (file, line);
    parse_line (line, biasesL5);
    getline (file, line);
    getline (file, line);
    parse_line (line, weightsL6);
    getline (file, line);
    getline (file, line);
    parse_line (line, biasesL6);
    getline (file, line);
    getline (file, line);
    parse_line (line, weightsL7);
    getline (file, line);
    getline (file, line);
    parse_line (line, biasesL7);

    file.close ();

    // Initialize model
    Model::add_layer (98, weightsL1, biasesL1);
    Model::add_layer (65, weightsL2, biasesL2);
    Model::add_layer (50, weightsL3, biasesL3);
    Model::add_layer (30, weightsL4, biasesL4);
    Model::add_layer (25, weightsL5, biasesL5);
    Model::add_layer (40, weightsL6, biasesL6);
    Model::add_layer (52, weightsL7, biasesL7);
    Model::add_layer (0, nullptr, nullptr);
}