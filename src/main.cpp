/**
 * @file main.cpp
 * @brief StartHack Hackathon HPC Neural Network on Digit Recognition Sponsered by QDX
 * @authors Carlvince Tan, Lucas Yu, Peter Lu, Volodymyr Kazmirchuk
 * @date 2024
 * @copyright University of Melbourne
*/

#include "reader.h"
#include "model.h"
#include "params.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

#define NOW start = chrono::high_resolution_clock::now ()
#define ELAPSED chrono::duration_cast <chrono::microseconds> (chrono::high_resolution_clock::now () - start).count () / 1000.0

using namespace std;


/**
 * Process all tensors in /tensors directory
*/
void process_directory (int repeats = 1) {
    string tensors_path = filesystem::current_path ().string () + "/tensors";

    // Get number of files in directory
    // string tpath = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    // int digits = tpath.size () - 8 - tensors_path.size ();
    // int size = 1; for (int i = 0; i < digits; i ++, size *= 10);
    auto it = filesystem::directory_iterator (tensors_path);
    int size = distance (it, {});
    int digits = 0; for (int i = size; i > 0; i /= 10, digits ++);
    char* aux = new char [size + 1];

    int batch = 10;
    Model model (batch);

    // Profiling
    long double avg = 0;
    for (int r = 0; r < repeats; r ++) {
        auto NOW;
        
        /*/ Processing every file in directory
        for (auto& entry : filesystem::directory_iterator (tensors_path)) {
            // Reading data and processing
            string path = entry.path ().string ();
            read_input (path, model.get_inputs ());
            int* out = model.forward_pass ();
            int res = out [0];

            // Converting and saving the result
            char letter = res % 2 ? char (97 + res / 2) : char (65 + res / 2);
            aux [stoi (path.substr (tensors_path.size () + 1, digits))] = letter;
        }*/

        int mapping [batch];
        auto it = filesystem::directory_iterator (tensors_path);
        for (int i = 0; i < size; it ++) {
            string path = (*it).path ().string ();
            read_input (path, model.get_inputs () + i % batch * INPUT);
            mapping [i % batch] = stoi (path.substr (tensors_path.size () + 1, digits));
            i ++;

            if (i % batch == 0 || i == size) {
                int s = i % batch == 0 ? batch : i % batch;
                int* out = model.sub_forward_pass (s);
                for (int u = 0; u < s; u ++) {
                    int res = out [u];
                    aux [mapping [u]] = res % 2 ? char (97 + res / 2) : char (65 + res / 2);
                }
            }
        }
        
        avg += ELAPSED;
        if (r % 100 == 1) { cout << "Completed " << r << " repeats" << endl; }
    }
    cout << "Average directory processing time: " << avg / repeats << " milliseconds" << endl;

    // Writing results to csv
    ofstream fout ("results.csv");
    fout.tie ();
    fout << "image number,label" << '\n';
    for (int i = 1; i < size; i ++) {
        fout << i << ',' << aux [i] << '\n';
    }
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
    process_directory (argc > 1 ? atoi (argv [1]) : 1);
    
    Model::free ();
    return 0;
}