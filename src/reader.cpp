#include "reader.h"
#include <fstream>
#include <sstream>

using namespace std;


// Allocate memory
Parameters::Parameters () {
    weightsL1 = new double [225*98];
    weightsL2 = new double [98*65];
    weightsL3 = new double [65*50];
    weightsL4 = new double [50*30];
    weightsL5 = new double [30*25];
    weightsL6 = new double [25*40];
    weightsL7 = new double [40*52];

    biasesL1 = new double [98];
    biasesL2 = new double [65];
    biasesL3 = new double [50];
    biasesL4 = new double [30];
    biasesL5 = new double [25];
    biasesL6 = new double [40];
    biasesL7 = new double [52];
}


// Parses a line into an array
void parseLine (const string& line, double* values) {
    int count = 0;
    istringstream stream (line);
    string token;

    while (getline (stream, token, ',')) {
        values [count ++] = stod (token);
    }
}


// Read values from file to an array
double* read_input (const string& filename) {
    ifstream file (filename);
    string line;
    getline (file, line);
    file.close ();

    double* arr = new double [225];
    parseLine (line, arr);
    return arr;
}


// Reads weights from file to parameters
void read_parameters(const string& filename, Parameters *parameters) {
    ifstream file (filename);
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