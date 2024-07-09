/**
 * @file main.cpp
 * @brief StartHack Hackathon HPC Neural Network on Alphabetical Character Recognition Sponsered by QDX
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
#include "helpers.h"

using namespace std;


#define MODEL_COUNT thread::hardware_concurrency ()
#define READER_COUNT thread::hardware_concurrency ()
#define BATCH_SIZE 1024


/**
 * Process all tensors in /tensors directory
*/
void process_directory (const string& tensors_path, int repeats = 1) {
    // Get the maximum number of files in directory
    string tpath = (*filesystem::directory_iterator (tensors_path)).path ().string ();
    int digits = tpath.size () - 8 - tensors_path.size (), cnt = 0;
    int size = 1; for (int i = 0; i < digits; i ++, size *= 10);
    char* aux = new char [size + 1];

    // Initialized parameters
    int model_count = MODEL_COUNT;
    tq free_models = tq ();
    int batch = BATCH_SIZE;
    for (int i = 0; i < model_count; i ++) {
        free_models.push (new Model (batch));
    }
    atomic_int free_readers = READER_COUNT;

    // Profiling
    long double avg = 0;
    for (int r = 0; r < repeats; r ++) {
        auto NOW;
        
        // Reset local values
        cnt = 0;
        int last_launch = 0;
        auto it = filesystem::directory_iterator (tensors_path);

        // Process every file in folder
        while (true) {
            string path = (*it).path ().string ();

            // Wait for a free model
            while (free_models.empty ()) {IDLE}
            Model* current = (Model*) free_models.front ();

            // If model input is fully assigned to readers, put it in a waiting mode and take the next one
            if (current -> is_ready ()) {
                free_models.pop ();
                thread t (&Model::forward_pass, current, aux, &free_models, 0);
                t.detach ();
                last_launch = cnt;
                
                while (free_models.empty ()) {IDLE}
                current = (Model*) free_models.front ();
            }

            // Wait for a free reader, then assign in to the part of current model input
            while (free_readers == 0) {IDLE}
            current -> process_input (path, stoi (path.substr (tensors_path.size () + 1, digits)), &free_readers);

            it ++; cnt ++;
            
            // It was the file in folder, launch current model if hasn't done it yet
            if (it == end (it)) {
                if (last_launch != cnt) {
                    free_models.pop ();
                    thread t (&Model::forward_pass, current, aux, &free_models, cnt - last_launch);
                    t.detach ();
                }
                break;
            }
        }

        // Wait for all models to finish processing
        while (free_models.size () != model_count) {IDLE}

        avg += ELAPSED;
        //if (r % 100 == 1) { cout << "Completed " << r << " repeats" << endl; }
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

    // Check correctness
    // int correct = 0;
    // for (int i = 1; i <= cnt; i ++) {
    //     int res = 1 + (aux [i] > 96 ? (aux [i] - 97) * 2 + 1 : (aux [i] - 65) * 2);
    //     if (res == (i - 1) % 52 + 1) { correct ++; }
    // }
    // cout << "Correct: " << correct << " out of " << cnt << endl;

    // Free allocated resources
    while (!free_models.empty ()) {
        delete (Model*) free_models.front ();
        free_models.pop ();
    }
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

    // Initialize parameters
    string wab = argc > 1 ? argv [1] : "weights_and_biases.txt";
    string tensors_path = argc > 2 ? argv [2] : filesystem::current_path ().string () + "/tensors";
    int repeats = argc > 3 ? atoi (argv [3]) : 1;

    // Initialize model
    auto NOW;
    Model::init ();
    init_model (wab);
    cout << "Model initialized in " << ELAPSED << " milliseconds" << endl;

    // Process directory
    process_directory (tensors_path, repeats);
    
    // Free resources
    Model::free ();
    
    return 0;
}
