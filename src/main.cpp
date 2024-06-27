/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *        StartHack Hackathon HPC Neural Network on Digit Recognition Sponsered by QDX       *
 *                                          Authors                                          *
 *                   Carlvince Tan, Lucas Yu, Peter Lu, Volodymyr Kazmirchuk                 *
 *                                                                                           *
 *                          (SHORT DESCRIPTION OF NEURAL NETWORK)                            *
 *                                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "reader.h"
#include "model.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;

// Converts a string to double*
double* str_to_a (const string& s, int expected) {
    double* arr = new double [expected];
    int pos = 0, begin = 0;
    for (int i = 0; i < int (s.size ()); i ++) {
        if (s [i] == ',') {
            arr [pos] = stod (s.substr (begin, i - begin));
            begin = i + 1;
            pos ++;
        }
    }
    arr [pos] = stod (s.substr (begin));
    return arr;
}

// Reads an input tensor to double*
double* bonk (string filename) {
    ifstream fin (filename);
    string s;
    getline (fin, s);
    fin.close ();
    return str_to_a (s, 225);
}

int main () {
    // Optimisations to try:
    // .tie, ios_base::sync_with_stdio, .flush

    /*// Create matrices and vectors
    vector<long double> inputVector;
    vector<vector<long double> > inputMatrix;

    vector<vector<long double> > weightsL1;
    vector<vector<long double> > weightsL2;
    vector<vector<long double> > weightsL3;
    vector<vector<long double> > weightsL4;
    vector<vector<long double> > weightsL5;
    vector<vector<long double> > weightsL6;
    vector<vector<long double> > weightsL7;

    //exmple of using new to declare dynamic arrs
    // double* var = new double [88];
    // delete[] var;

    vector<long double> biasesL1;
    vector<long double> biasesL2;
    vector<long double> biasesL3;
    vector<long double> biasesL4;
    vector<long double> biasesL5;
    vector<long double> biasesL6;
    vector<long double> biasesL7;

    // Parse Input Tensor
    Parser tensorParser("tensors/01out.txt");

    // Parse to Vector and Matrix
    tensorParser.parseToVector(inputVector);
    tensorParser.parseToMatrix(inputMatrix, 15, 15);

    // Parse Weights and Biases
    Parser weightsParser("weights_and_biases.txt");

    // Parse Weights
    weightsParser.parseWeights(weightsL1, 1, 225, 98);
    weightsParser.parseWeights(weightsL2, 2, 98, 65);
    weightsParser.parseWeights(weightsL3, 3, 65, 50);
    weightsParser.parseWeights(weightsL4, 4, 50, 30);
    weightsParser.parseWeights(weightsL5, 5, 30, 25);
    weightsParser.parseWeights(weightsL6, 6, 25, 40);
    weightsParser.parseWeights(weightsL7, 7, 40, 52);

    // Parse Biases
    weightsParser.parseBiases(biasesL1, 1);
    weightsParser.parseBiases(biasesL2, 2);
    weightsParser.parseBiases(biasesL3, 3);
    weightsParser.parseBiases(biasesL4, 4);
    weightsParser.parseBiases(biasesL5, 5);
    weightsParser.parseBiases(biasesL6, 6);
    weightsParser.parseBiases(biasesL7, 7);*/

    // Initialize model
    Model model (7, 225);
    ifstream fmodel ("weights_and_biases.txt");
    string weights, biases;
    getline (fmodel, weights); getline (fmodel, weights); getline (fmodel, biases); getline (fmodel, biases);
    model.add_layer (98, str_to_a (weights, 225 * 98), str_to_a (biases, 98));
    getline (fmodel, weights); getline (fmodel, weights); getline (fmodel, biases); getline (fmodel, biases);
    model.add_layer (65, str_to_a (weights, 98 * 65), str_to_a (biases, 65));
    getline (fmodel, weights); getline (fmodel, weights); getline (fmodel, biases); getline (fmodel, biases);
    model.add_layer (50, str_to_a (weights, 65 * 50), str_to_a (biases, 50));
    getline (fmodel, weights); getline (fmodel, weights); getline (fmodel, biases); getline (fmodel, biases);
    model.add_layer (30, str_to_a (weights, 50 * 30), str_to_a (biases, 30));
    getline (fmodel, weights); getline (fmodel, weights); getline (fmodel, biases); getline (fmodel, biases);
    model.add_layer (25, str_to_a (weights, 25 * 30), str_to_a (biases, 25));
    getline (fmodel, weights); getline (fmodel, weights); getline (fmodel, biases); getline (fmodel, biases);
    model.add_layer (40, str_to_a (weights, 40 * 30), str_to_a (biases, 40));
    getline (fmodel, weights); getline (fmodel, weights); getline (fmodel, biases); getline (fmodel, biases);
    model.add_layer (52, str_to_a (weights, 52 * 40), str_to_a (biases, 52));
    
    // Dark magic
    string tensors_path = filesystem::current_path ().string () + "/tensors";
    string path = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    int digits = path.size () - 8 - tensors_path.size ();
    int size = 1;
    for (int i = 0; i < digits; i ++, size *= 10);
    char* aux = new char [size];

    // Processing every file in directory
    int cnt = 1;
    for (auto& entry : filesystem::directory_iterator (tensors_path)) {
        // Reading data and processing
        string path = entry.path ().string ();
        int res = model.forward_pass (bonk (path));

        // Converting and saving the result
        char letter = res % 2 ? char (97 + res / 2) : char (65 + res / 2);
        aux [stoi (path.substr (tensors_path.size () + 1, digits))] = letter;
        cnt ++;
    }

    // Writing results to csv
    ofstream fout ("results.csv");
    fout << "image number,label" << '\n';
    for (int i = 1; i < cnt; i ++) {
        fout << i << ',' << aux [i] << '\n';
        cout << aux [i] << ' ';
    } cout << endl;
    fout.close ();
    delete[] aux;

    return 0;
}