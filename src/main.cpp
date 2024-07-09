/**
 * @file main.cpp
 * @brief StartHack Hackathon HPC Neural Network on Digit Recognition Sponsered by QDX
 * @authors Carlvince Tan, Lucas Yu, Peter Lu, Volodymyr Kazmirchuk
 * @date 2024
 * @copyright University of Melbourne
*/

#include "eigenreader.h"
#include "eigenmodel.h"
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

//#include <Eigen/Dense>

/**
 * The main function
*/
int main (int argc, char* argv []) {
    std::ios_base::sync_with_stdio (false);

    //initialising parameters
    std::vector<MatrixXd> weights = init_weights();
    std::vector<MatrixXd> biases = init_biases();

    //reading in weights and biases
    read_weights_biases("weights_and_biases.txt", &weights, &biases);

    std::string tensor_path = std::filesystem::current_path ().string () + "/tensors_10k";
    // reading inputs into batches
    std::vector<input_batch> batches = read_inputs(tensor_path);

    // passes in batches individuall, decrease batch_size, this part is for parallelisation
    int num_input_files = collect_file_paths(tensor_path).size()+1;
    char *final_output = new char[num_input_files];
    for (size_t i=0; i<batches.size(); i++) {
        // passes inputs through all 7 layers of neural net, outputs raw output
        MatrixXd output = pass_through_model(batches[i].inputs, weights, biases);
        // process output and returns mapping of file number to character
        struct output_batch results = create_output_map(batches[i].input_number, &output);

        // writing to final output
        for (int i=0; i<BATCH_SIZE; i++) {
            final_output[results.input_numbers[i]] = results.character[i];
            //std::cout << "input file: " << results.input_numbers[i] << " maxindex:" << results.character[i] << "\n";
        }
    }

    // writing final output to results.csv
    std::ofstream fout("results.csv");
    fout.tie ();
    fout << "image number,label" << '\n';
    for (int i = 1; i < num_input_files; i ++) {
        fout << i << ',' << final_output[i] << '\n';
        //cout << aux [i] << ' ';
    }
    //cout << endl;
    fout.flush ();
    fout.close ();
    delete[] final_output;

    return 0;
}
