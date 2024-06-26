/*
 * Functions that parse input array to either Vector or Matrix
 */
#include "parser.h"

// Constructor definition
Parser::Parser(string fileName) {
    FILE_NAME = fileName;
}

void Parser::parseToVector(vector<long double> &vector) {
    ifstream file(FILE_NAME);

    string line;
    getline(file, line);

    stringstream stream(line);
    string token;

    // Parse long doubles into vector
    while (getline(stream, token, ',')) {
        vector.push_back(stold(token));
    }

    file.close();
}

void Parser::parseToMatrix(vector<vector<long double> > &matrix, int dimX, int dimY) {
    ifstream file(FILE_NAME);

    string line;
    getline(file, line);

    stringstream stream(line);
    string token;

    // Initialize matrix with dimensions
    matrix.resize(dimY, vector<long double>(dimX));

    // Parse long doubles into matrix
    int count = 0;
    while (getline(stream, token, ',')) {
        matrix[count / dimX][count % dimX] = stold(token);
        count++;
    }

    file.close();
}

void Parser::parseWeights(vector<vector<long double> > &weights, int layer, int dimIn, int dimOut) {
    ifstream file(FILE_NAME);

    string line;
    while (getline(file, line)) {
        if (regex_match(line, regex("fc" + to_string(layer) + R"(.*\.weight:$)"))) {
            // Read in next line
            getline(file, line);

            stringstream stream(line);
            string token;

            // Initialize weights matrix with dimensions
            weights.resize(dimOut, vector<long double>(dimIn));

            // Parse long doubles into matrix
            int count = 0;
            while (getline(stream, token, ',')) {
                weights[count / dimIn][count % dimIn] = stold(token);
                count++;
            }
        }
    }

    file.close();
}

<<<<<<< HEAD
void Parser::parseBiases(vector<long double> &biases, int layer) {
=======
void Parser::parseBiases(vector<long double> &biases, int layer, int size) {
>>>>>>> ddb0dbe (Recommit)
    ifstream file(FILE_NAME);

    string line;
    while (getline(file, line)) {
        if (regex_match(line, regex("fc" + to_string(layer) + R"(.*\.bias:$)"))) {
            // Read in next line
            getline(file, line);

            stringstream stream(line);
            string token;

<<<<<<< HEAD
            // Parse long doubles into vector
            while (getline(stream, token, ',')) {
                biases.push_back(stold(token));
=======
            // Initialize biases vector with size
            biases.resize(size);

            // Parse long doubles into vector
            int count = 0;
            while (getline(stream, token, ',')) {
                biases[count++] = stold(token);
>>>>>>> ddb0dbe (Recommit)
            }
        }
    }

    file.close();
}
