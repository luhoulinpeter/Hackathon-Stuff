/**
 * @file main.cpp
 * @brief StartHack Hackathon HPC Neural Network on Digit Recognition Sponsered by QDX
 * @authors Carlvince Tan, Lucas Yu, Peter Lu, Volodymyr Kazmirchuk
 * @date 2024
 * @copyright University of Melbourne
*/

#include "reader.h"
#include "model.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

#define NOW start = chrono::high_resolution_clock::now ()
#define ELAPSED chrono::duration_cast <chrono::microseconds> (chrono::high_resolution_clock::now () - start).count () / 1000.0

using namespace std;

/**
 * Model Initialisation
*/
void init_model (Model& model) {
    // Read weights and biases
    Parameters* parameters = new Parameters;
    read_parameters ("weights_and_biases.txt", parameters);

    // Initialize model
    model.add_layer (98, parameters->weightsL1, parameters->biasesL1);
    model.add_layer (65, parameters->weightsL2, parameters->biasesL2);
    model.add_layer (50, parameters->weightsL3, parameters->biasesL3);
    model.add_layer (30, parameters->weightsL4, parameters->biasesL4);
    model.add_layer (25, parameters->weightsL5, parameters->biasesL5);
    model.add_layer (40, parameters->weightsL6, parameters->biasesL6);
    model.add_layer (52, parameters->weightsL7, parameters->biasesL7);

    delete parameters;
}


/**
 * Process all tensors in /tensors directory
*/
void process_directory (Model& model, int repeats = 1) {
    // Dark magic
    string tensors_path = filesystem::current_path ().string () + "/tensors";
    string path = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    int digits = path.size () - 8 - tensors_path.size ();
    int size = 1;
    for (int i = 0; i < digits; i ++, size *= 10);
    char* aux = new char [size];
    int cnt;

    // Profiling
    long double avg = 0;
    for (int i = 0; i < repeats; i ++) {
        auto NOW;

        // Processing every file in directory
        cnt = 1;
        for (auto& entry : filesystem::directory_iterator (tensors_path)) {
            // Reading data and processing
            string path = entry.path ().string ();
            int res = model.forward_pass (read_input (path));

            // Converting and saving the result
            char letter = res % 2 ? char (97 + res / 2) : char (65 + res / 2);
            aux [stoi (path.substr (tensors_path.size () + 1, digits))] = letter;
            cnt ++;
        }
        
        avg += ELAPSED;
        if (i % 100 == 1) { cout << "Completed " << i << " repeats" << endl; }
    }
    cout << "Average directory processing time: " << avg / repeats << " milliseconds" << endl;

    // Writing results to csv
    ofstream fout ("results.csv");
    fout.tie ();
    fout << "image number,label" << '\n';
    for (int i = 1; i < cnt; i ++) {
        fout << i << ',' << aux [i] << '\n';
        //cout << aux [i] << ' ';
    }
    //cout << endl;
    fout.flush ();
    fout.close ();
    delete[] aux;
}


// Optimisations to try
// IO:      ios_base::sync_with_stdio, .tie, .flush     - Done, +
// Stings:  reconsider getline                          - On hold
// Math:    multiple inputs, simpler exp
// General: malloc vs new, multiprocessing

/**
 * Single-Thread execution of Model Inference
*/
int main () {
    ios_base::sync_with_stdio (false);

    // Initialize model
    auto NOW;
    Model model (7, 225);
    init_model (model);
    cout << "Model initialized in " << ELAPSED << " milliseconds" << endl;

    // Process directory (avg)
    process_directory (model, 500);
    // NOW;
    // process_directory (model);
    // cout << "Done in " << ELAPSED << " milliseconds" << endl;
    
    return 0;
}