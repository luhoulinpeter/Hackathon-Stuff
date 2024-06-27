/*
 * Functions that read input array to either Vector or Matrix
 */
#include "reader.h"

// Constructor definition
Reader::Reader(string fileName) {
    FILE_NAME = fileName;
}

void Reader::readInput(double* vector) {
    ifstream file(FILE_NAME);

    string line;
    getline(file, line);

    istringstream stream(line);
    string token;

    // Reset the stream and allocate memory
    vector = (double*)malloc(225 * sizeof(double));

    // Read long doubles into vector
    int i = 0;
    while (getline(stream, token, ',')) {
        vector[i++] = stod(token);
    }

    file.close();
}

void parseLine(const string line, double* values) {
    int count = 0;
    istringstream stream(line);
    string token;

    while (getline(stream, token, ',')) {
        values[count++] = stod(token);
    }
}

void Reader::readParameters(Parameters *parameters) {
    ifstream file(FILE_NAME);
    string line;
    
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->weightsL1);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->biasesL1);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->weightsL2);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->biasesL2);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->weightsL3);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->biasesL3);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->weightsL4);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->biasesL4);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->weightsL5);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->biasesL5);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->weightsL6);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->biasesL6);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->weightsL7);
    getline(file, line);
    getline(file, line);
    parseLine(line, parameters->biasesL7);

    file.close();
}