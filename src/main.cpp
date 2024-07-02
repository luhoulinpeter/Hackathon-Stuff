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
#include <omp.h>
#include <vector>

#define NOW start = chrono::high_resolution_clock::now ()
#define ELAPSED chrono::duration_cast <chrono::microseconds> (chrono::high_resolution_clock::now () - start).count () / 1000.0

using namespace std;


/**
 * Model initialisation
*/
void init_model () {
    // Read weights and biases
    Parameters* parameters = read_parameters ("weights_and_biases.txt");

    // Initialize model
    Model::add_layer (98, parameters -> weightsL1, parameters -> biasesL1);
    Model::add_layer (65, parameters -> weightsL2, parameters -> biasesL2);
    Model::add_layer (50, parameters -> weightsL3, parameters -> biasesL3);
    Model::add_layer (30, parameters -> weightsL4, parameters -> biasesL4);
    Model::add_layer (25, parameters -> weightsL5, parameters -> biasesL5);
    Model::add_layer (40, parameters -> weightsL6, parameters -> biasesL6);
    Model::add_layer (52, parameters -> weightsL7, parameters -> biasesL7);

    delete parameters;
}


// Function to collect all file paths in the directory
vector<string> collect_file_paths(const string& directory) {
    vector<string> file_paths;
    #pragma omp parallel for ordered
    for (const auto& entry : filesystem::directory_iterator(directory)) {
        #pragma omp ordered
        file_paths.push_back(entry.path().string());
    }
    return file_paths;
}


/**
 * Process all tensors in /tensors directory
*/
void process_directory (Model& model, int repeats = 1) {
    // Dark magic
    string tensors_path = filesystem::current_path ().string () + "/tensors";
    vector<string> file_paths = collect_file_paths(tensors_path);
    if (file_paths.empty()) {
        cerr << "No files found in the tensors directory." << endl;
        return;
    }
    // string path = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    string path = file_paths[0];

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
        cnt=1;
    
        #pragma omp parallel for
        // for (auto& entry : filesystem::directory_iterator (tensors_path)) {
        for (size_t j = 0; j < file_paths.size(); j++) {
            // Reading data and processing
            // string path = entry.path ().string ();
            string path = file_paths[j];
            int* out = model.forward_pass (read_input (path));
            int res = out [0];
            delete[] out;

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
// Math:    multiple inputs (done, needs testing), faster exp
// Mp:      multiprocessing vs threading

/**
 * The main function
*/
int main (int argc, char* argv []) {
    ios_base::sync_with_stdio (false);

    // Initialize model
    auto NOW;
    Model::init ();
    init_model ();
    cout << "Model initialized in " << ELAPSED << " milliseconds" << endl;

    // Process directory (avg)
    Model model (1);
    process_directory (model, argc > 1 ? atoi (argv [1]) : 1);
    
    Model::free ();
    return 0;
}