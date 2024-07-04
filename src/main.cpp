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
    string tpath = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    int digits = tpath.size () - 8 - tensors_path.size (), cnt = 0;
    int size = 1; for (int i = 0; i < digits; i ++, size *= 10);
    char* aux = new char [size + 1];

    int batch = 10;
    Model model (batch);

    // Profiling
    long double avg = 0;
    for (int r = 0; r < repeats; r ++) {
        auto NOW;
        
        int mapping [batch];
        cnt = 0;
        auto it = filesystem::directory_iterator (tensors_path);
        while (true) {
            string path = (*it).path ().string ();
            read_input (path, model.get_inputs () + cnt % batch * INPUT);
            mapping [cnt % batch] = stoi (path.substr (tensors_path.size () + 1, digits));
            it ++; cnt ++;

            if (cnt % batch == 0 || it == end (it)) {
                int s = cnt % batch == 0 ? batch : cnt % batch;
                int* out = model.forward_pass (s);
                for (int u = 0; u < s; u ++) {
                    int res = out [u];
                    aux [mapping [u]] = res % 2 ? char (97 + res / 2) : char (65 + res / 2);
                }
                if (it == end (it)) { break; }
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
    for (int i = 1; i <= cnt; i ++) {
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