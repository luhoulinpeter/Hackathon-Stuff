#include "reader.h"
#include "params.h"
#include <fstream>
#include <sstream>

using namespace std;

/**
 * Parameters constructor
 * Allocates memory
 */
Parameters::Parameters () {
    weightsL1 = new double [INPUT*L1];
    weightsL2 = new double [L1*L2];
    weightsL3 = new double [L2*L3];
    weightsL4 = new double [L3*L4];
    weightsL5 = new double [L4*L5];
    weightsL6 = new double [L5*L6];
    weightsL7 = new double [L6*L7];

    biasesL1 = new double [L1];
    biasesL2 = new double [L2];
    biasesL3 = new double [L3];
    biasesL4 = new double [L4];
    biasesL5 = new double [L5];
    biasesL6 = new double [L6];
    biasesL7 = new double [L7];
}


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
 * Takes a filename to read an input from and returns an array of values
 */
double* read_input (const string& filename) {
    ifstream file (filename);
    string line;
    getline (file, line);
    file.close ();

    double* arr = new double [INPUT];
    parse_line (line, arr);
    return arr;
}


/**
 * Read weights and biases from file to parameters
 * Takes a weights filename and returns a Parameters structure with parsed values
 */
Parameters* read_parameters (const string& filename) {
    
   Parameters* parameters = new Parameters;
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    string fileContent((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    istringstream stream(fileContent);
    string line;
    auto parse_and_advance = [&stream, &line](double* values) {
        getline(stream, line);  // Skip the line before the data
        getline(stream, line);
        parse_line(line, values);
    };

    parse_and_advance(parameters->weightsL1);
    parse_and_advance(parameters->biasesL1);
    parse_and_advance(parameters->weightsL2);
    parse_and_advance(parameters->biasesL2);
    parse_and_advance(parameters->weightsL3);
    parse_and_advance(parameters->biasesL3);
    parse_and_advance(parameters->weightsL4);
    parse_and_advance(parameters->biasesL4);
    parse_and_advance(parameters->weightsL5);
    parse_and_advance(parameters->biasesL5);
    parse_and_advance(parameters->weightsL6);
    parse_and_advance(parameters->biasesL6);
    parse_and_advance(parameters->weightsL7);
    parse_and_advance(parameters->biasesL7);

    return parameters;

}