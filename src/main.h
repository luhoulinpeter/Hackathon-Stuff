#ifndef MAIN_HEADER_H
#define MAIN_HEADER_H

#include "reader.h"
#include "model.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;

// Define the Parameters struct
typedef struct {
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
} Parameters;

#endif 
