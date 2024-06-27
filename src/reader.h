#ifndef READER_H
#define READER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <string>
#include <vector>
#include "main.h"

using namespace std;

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
