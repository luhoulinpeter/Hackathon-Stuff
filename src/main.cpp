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
#include <sstream>
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
void init_model_vova (Model& model) {
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


// Initializes model
void init_model_carlvince (Model& model) {
    Parameters *parameters = (Parameters *)malloc(sizeof(Parameters));

    // Allocate memory
    parameters->weightsL1 = new double[22050];
    parameters->weightsL2 = new double[6370];
    parameters->weightsL3 = new double[3250];
    parameters->weightsL4 = new double[1500];
    parameters->weightsL5 = new double[750];
    parameters->weightsL6 = new double[1000];
    parameters->weightsL7 = new double[2080];

    parameters->biasesL1 = new double[98];
    parameters->biasesL2 = new double[65];
    parameters->biasesL3 = new double[50];
    parameters->biasesL4 = new double[30];
    parameters->biasesL5 = new double[25];
    parameters->biasesL6 = new double[40];
    parameters->biasesL7 = new double[52];

    // read Weights and Biases
    Reader weightsReader("weights_and_biases.txt");
    weightsReader.readParameters(parameters);

    // Initialize model
    model.add_layer(98, parameters->weightsL1, parameters->biasesL1);
    model.add_layer(65, parameters->weightsL2, parameters->biasesL2);
    model.add_layer(50, parameters->weightsL3, parameters->biasesL3);
    model.add_layer(30, parameters->weightsL4, parameters->biasesL4);
    model.add_layer(25, parameters->weightsL5, parameters->biasesL5);
    model.add_layer(40, parameters->weightsL6, parameters->biasesL6);
    model.add_layer(52, parameters->weightsL7, parameters->biasesL7);

    free(parameters);
}


// Processes all tensors in a directory
void process_directory (Model& model, bool v) {
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
        int res;
        if (v) {
            res = model.forward_pass (bonk (path));
        }
        else {
            double* input = new double [225];
            Reader tensorReader (path);
            tensorReader.readInput (input);
            res = model.forward_pass (input);
        }

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
    init_model_vova (model);
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
        process_directory (model, true);
        avg += elapsed;
        if (i % 100 == 0) { cout << "Completed " << i << " repeats" << endl; }
    }
    cout << "Average directory processing time: " << avg / repeats << " milliseconds" << endl;
    
    return 0;
}