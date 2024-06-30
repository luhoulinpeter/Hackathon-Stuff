#ifndef READER_H
#define READER_H

#include <string>

#define LAYERS 7
#define INPUT 225
#define L1 98
#define L2 65
#define L3 50
#define L4 30
#define L5 25
#define L6 40
#define L7 52

// Parameters struct
struct Parameters {
    double* weightsL1;
    double* weightsL2;
    double* weightsL3;
    double* weightsL4;
    double* weightsL5;
    double* weightsL6;
    double* weightsL7;

    double* biasesL1;
    double* biasesL2;
    double* biasesL3;
    double* biasesL4;
    double* biasesL5;
    double* biasesL6;
    double* biasesL7;

    Parameters ();
};

// Read values from file to an array
double* read_input (const std::string& filename);

// Reads weights from file to parameters
Parameters* read_parameters (const std::string& filename);

#endif 
