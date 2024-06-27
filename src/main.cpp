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
#include "main.h"
#include <iostream>
#include <filesystem>
#include <chrono>

#define elapsed chrono::duration_cast <chrono::microseconds> (chrono::system_clock::now () - start).count () / 1000.0

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


// Initializes model
void init_model (Model& model) {
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
    fmodel.close ();
}


// Processes all tensors in a directory
void process_directory (Model& model) {
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
        //cout << aux [i] << ' ';
    }
    //cout << endl;
    fout.close ();
    delete[] aux;
}


int main () {
    // Optimisations to try:
    // .tie, ios_base::sync_with_stdio, .flush

    // Initialize model
    auto start = chrono::system_clock::now ();
    Model model (7, 225);
    init_model (model);
    cout << "Model initialized in " << elapsed << " milliseconds" << endl;

    // Process directory (once)
    // start = chrono::system_clock::now ();
    // process_directory (model);
    // cout << "Directory processed in " << elapsed << " milliseconds" << endl;

    // Process directory (avg)
    int repeats = 1000;
    long double avg = 0;
    for (int i = 0; i < repeats; i ++) {
        start = chrono::system_clock::now ();
        process_directory (model);
        avg += elapsed;
        if (i % 100 == 0) { cout << "Completed " << i << " repeats" << endl; }
    }
    cout << "Average directory processing time: " << avg / repeats << " milliseconds" << endl;
    
    /*-----------------------------------------------------------------------------------------------*/
    /*                                     MY PART                                                   */
    /*-----------------------------------------------------------------------------------------------*/
    Parameters *parameters = (Parameters *)malloc(sizeof(Parameters));

    // Allocate memory
    parameters->weightsL1 = (double *)malloc(22050 * sizeof(double));
    parameters->weightsL2 = (double *)malloc(6370 * sizeof(double));
    parameters->weightsL3 = (double *)malloc(3250 * sizeof(double));
    parameters->weightsL4 = (double *)malloc(1500 * sizeof(double));
    parameters->weightsL5 = (double *)malloc(750 * sizeof(double));
    parameters->weightsL6 = (double *)malloc(1000 * sizeof(double));
    parameters->weightsL7 = (double *)malloc(2080 * sizeof(double));

    parameters->biasesL1 = (double *)malloc(98 * sizeof(double));
    parameters->biasesL2 = (double *)malloc(65 * sizeof(double));
    parameters->biasesL3 = (double *)malloc(50 * sizeof(double));
    parameters->biasesL4 = (double *)malloc(30 * sizeof(double));
    parameters->biasesL5 = (double *)malloc(25 * sizeof(double));
    parameters->biasesL6 = (double *)malloc(40 * sizeof(double));
    parameters->biasesL7 = (double *)malloc(52 * sizeof(double));

    // read Weights and Biases
    Reader weightsReader("weights_and_biases.txt");
    weightsReader.readParameters(parameters);

    // Initialize model
    Model model (7, 225);
    // What a nice temporary solution!
    model.add_layer (98, parameters->weightsL1, parameters->biasesL1);
    model.add_layer (65, parameters->weightsL2, parameters->biasesL2);
    model.add_layer (50, parameters->weightsL3, parameters->biasesL3);
    model.add_layer (30, parameters->weightsL4, parameters->biasesL4);
    model.add_layer (25, parameters->weightsL5, parameters->biasesL5);
    model.add_layer (40, parameters->weightsL6, parameters->biasesL6);
    model.add_layer (52, parameters->weightsL7, parameters->biasesL7);
    
    // Dark magic
    string tensors_path = filesystem::current_path ().string () + "/tensors";
    string path = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    int digits = path.size () - 8 - tensors_path.size ();
    int size = 1;
    for (int i = 0; i < digits; i ++, size *= 10);
    char* aux = new char [size];

    // Processing every file
    int cnt = 1;
    for (auto& entry : filesystem::directory_iterator (tensors_path)) {
        // Reading data and processing
        string path = entry.path ().string ();
        double *input = (double *)malloc(225 * sizeof(double));
        Reader tensorReader (path);
        tensorReader.readInput (input);
        int res = model.forward_pass (input);

        // Converting and saving the result
        char letter = res % 2 ? char (97 + res / 2) : char (65 + res / 2);
        string substr = path.substr (tensors_path.size () + 1, digits);
        aux [stoi (substr)] = letter;
        cout << "For \'" << substr <<  "\' the result is " << letter << '\n';
        cnt ++;
    }
    cout << aux << '\n';

    // Writing results to csv
    ofstream fout ("results.csv");
    fout << "image number,label" << '\n';
    for (int i = 1; i <= cnt; i ++) {
        fout << i << ',' << aux [i] << '\n';
    }
    fout.close ();
    delete[] aux;

    // Clear Memory

    /*-----------------------------------------------------------------------------------------------*/
    /*-----------------------------------------------------------------------------------------------*/
    /*-----------------------------------------------------------------------------------------------*/

    //exmple of using new to declare dynamic arrs
    // double* var = new double [88];
    // delete[] var;

    // vector<long double> biasesL1;
    // vector<long double> biasesL2;
    // vector<long double> biasesL3;
    // vector<long double> biasesL4;
    // vector<long double> biasesL5;
    // vector<long double> biasesL6;
    // vector<long double> biasesL7;

    // // Parse Input Tensor
    // Parser tensorParser("tensors/01out.txt");

    // // Parse to Vector and Matrix
    // tensorParser.parseToVector(inputVector);
    // tensorParser.parseToMatrix(inputMatrix, 15, 15);

    // // Parse Weights and Biases
    // Parser weightsParser("weights_and_biases.txt");

    // // Parse Weights
    // weightsParser.parseWeights(weightsL1, 1, 225, 98);
    // weightsParser.parseWeights(weightsL2, 2, 98, 65);
    // weightsParser.parseWeights(weightsL3, 3, 65, 50);
    // weightsParser.parseWeights(weightsL4, 4, 50, 30);
    // weightsParser.parseWeights(weightsL5, 5, 30, 25);
    // weightsParser.parseWeights(weightsL6, 6, 25, 40);
    // weightsParser.parseWeights(weightsL7, 7, 40, 52);

    // // Parse Biases
    // weightsParser.parseBiases(biasesL1, 1);
    // weightsParser.parseBiases(biasesL2, 2);
    // weightsParser.parseBiases(biasesL3, 3);
    // weightsParser.parseBiases(biasesL4, 4);
    // weightsParser.parseBiases(biasesL5, 5);
    // weightsParser.parseBiases(biasesL6, 6);
    // weightsParser.parseBiases(biasesL7, 7);

    return 0;
}