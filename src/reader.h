#ifndef READER_H
#define READER_H

#include <string>

// Define the Parameters struct
struct Parameters {
    double *weightsL1;
    double *weightsL2;
    double *weightsL3;
    double *weightsL4;
    double *weightsL5;
    double *weightsL6;
    double *weightsL7;

    double *biasesL1;
    double *biasesL2;
    double *biasesL3;
    double *biasesL4;
    double *biasesL5;
    double *biasesL6;
    double *biasesL7;

    Parameters ();
};

// Read values from file to an array
double* read_input (const std::string& filename);

// Reads weights from file to parameters
void read_parameters (const std::string& filename, Parameters* parameters);

#endif 
