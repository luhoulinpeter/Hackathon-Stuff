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
#include "tq.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <atomic>
#include <chrono>

#define NOW start = chrono::high_resolution_clock::now ()
#define ELAPSED chrono::duration_cast <chrono::microseconds> (chrono::high_resolution_clock::now () - start).count () / 1000.0

using namespace std;


/**
 * Process all tensors in /tensors directory
*/
void process_directory (int repeats = 1) {
    string tensors_path = filesystem::current_path ().string () + "/tensors_10k";

    // Get number of files in directory
    string tpath = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    int digits = tpath.size () - 8 - tensors_path.size (), cnt = 0;
    int size = 1; for (int i = 0; i < digits; i ++, size *= 10);
    char* aux = new char [size + 1];

    int model_count = 4;
    tq free_models = tq ();
    int batch = 256;
    for (int i = 0; i < model_count; i ++) {
        free_models.push (new Model (batch));
    }
    atomic_int free_readers = 4;

    // Profiling
    long double avg = 0;
    for (int r = 0; r < repeats; r ++) {
        auto NOW;
        
        //int mapping [batch];
        cnt = 0;
        int last_launch = 0;
        auto it = filesystem::directory_iterator (tensors_path);
        while (true) {
            string path = (*it).path ().string ();

            while (free_models.empty ()) {}
            Model* current = (Model*) free_models.front ();

            if (current -> is_ready ()) {
                free_models.pop ();
                thread t (&Model::forward_pass, current, aux, &free_models, 0);
                t.detach ();
                last_launch = cnt;
                
                while (free_models.empty ()) {}
                current = (Model*) free_models.front ();
            }

            while (free_readers == 0) {}
            current -> process_input (path, stoi (path.substr (tensors_path.size () + 1, digits)), &free_readers);

            it ++; cnt ++;
            
            if (it == end (it)) {
                if (last_launch != cnt) {
                    free_models.pop ();
                    thread t (&Model::forward_pass, current, aux, &free_models, cnt - last_launch);
                    t.detach ();
                }
                break;
            }
        }
        while (free_models.size () != model_count) { this_thread::sleep_for (chrono::milliseconds (10)); }

        avg += ELAPSED;
        if (r % 100 == 1) { cout << "Completed " << r << " repeats" << endl; }
    }
    cout << "Average directory processing time: " << avg / repeats << " milliseconds" << endl;

    // Writing results to csv
    // ofstream fout ("results.csv");
    // fout.tie ();
    // fout << "image number,label" << '\n';
    // for (int i = 1; i <= 55; i ++) {
    //     fout << i << ',' << aux [i] << '\n';
    // }
    // fout.flush ();
    // fout.close ();

    // Check correctness
    int correct = 0;
    for (int i = 1; i <= cnt; i ++) {
        //res % 2 ? char (97 + res / 2) : char (65 + res / 2)
        int res = 1 + (aux [i] > 96 ? (aux [i] - 97) * 2 + 1 : (aux [i] - 65) * 2);
        if (res == (i - 1) % 52 + 1) { correct ++; }
    }
    cout << "Correct: " << correct << " out of " << cnt << endl;

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