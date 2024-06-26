#ifndef READER_H
#define READER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <string>
#include <vector>

using namespace std;

class Parser {
private:
    string FILE_NAME;

public:
    // Constructor
    Parser(string fileName);

    // Function: Parses values from file to a vector
    void parseToVector(vector<long double> &vector);

    // Function: Parses values from file to a matrix
    void parseToMatrix(vector<vector<long double> > &matrix, int dimX, int dimY);

    // Function: Parses Weights from a file to a matrix
    void parseWeights(vector<vector<long double> > &weights, int layer, int dimIn, int dimOut);

    // Function: Parses Biases from a file to a vector
    void parseBiases(vector<long double> &biases, int layer);
};

#endif // READER_H
