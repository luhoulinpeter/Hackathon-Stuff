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


// Initializes model
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


// Processes all tensors in a directory
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
        auto start = chrono::system_clock::now ();

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
        
        avg += elapsed;
        if (i % 100 == 0) { cout << "Completed " << i << " repeats" << endl; }
    }
    cout << "Average directory processing time: " << avg / repeats << " milliseconds" << endl;

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


// Optimisations to try
// IO:      ios_base::sync_with_stdio, .tie, .flush
// Stings:  reconsider getline
// Math:    multiple inputs, simpler exp
// General: malloc vs new, multiprocessing
int main () {

    // Initialize model
    auto start = chrono::system_clock::now ();
    Model model (7, 225);
    init_model (model);
    cout << "Model initialized in " << elapsed << " milliseconds" << endl;

    // Process directory (avg)
    process_directory (model, 500);
    
    return 0;
}