#ifndef READER_H
#define READER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <string>
#include <vector>

using namespace std;

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
};

class Reader {
private:
    string FILE_NAME;

public:
    // Constructor
    Reader(string fileName);

    // Function: reads values from file to a vector
    void readInput(double* vector);

    // Function: reads Weights from a file to a matrix
    void readParameters(Parameters *parameters);
};

#endif 
