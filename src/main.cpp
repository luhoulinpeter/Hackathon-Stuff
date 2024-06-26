/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * StartHack Hackathon HPC Neural Network on Digit Recognition Sponsered by QDX  *
 *                                   Authors                                     *
 *                      Carlvince, Lucas, Peter, Volodmyr                        *
 *                                                                               *
 *                    (SHORT DESCRIPTION OF NEURAL NETWORK)                      *
 *                                                                               *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Libraries
 */

#include <iostream>
#include <vector>
#include "input.h"
#include "parser.h"

int main () {
    // setup configs and constants such as:
    //   model parameters, number of images in folder, etc
    const int NUM_IMAGES = 5;

    // init model
    read_weights ("weights_and_biases.txt");

    // process each image
    
    for (int i = 0; i < NUM_IMAGES; i ++) {
        read_bitmap ("img" + std::to_string (i) + ".bmp");
    }

    /* Create matrices and vectors */
    vector<long double> inputVector {};
    vector<vector<long double> > inputMatrix {};

    vector<vector<long double> > weightsL1 {};
    vector<vector<long double> > weightsL2 {};
    vector<vector<long double> > weightsL3 {};
    vector<vector<long double> > weightsL4 {};
    vector<vector<long double> > weightsL5 {};
    vector<vector<long double> > weightsL6 {};
    vector<vector<long double> > weightsL7 {};

    vector<long double> biasesL1 {};
    vector<long double> biasesL2 {};
    vector<long double> biasesL3 {};
    vector<long double> biasesL4 {};
    vector<long double> biasesL5 {};
    vector<long double> biasesL6 {};
    vector<long double> biasesL7 {};

    /* Parse Input Tensor */
    Parser inputParser("/tensors/01out.txt"); // Use "\\tensors\\01out.txt" for Windows

    // Parse Inputs
    inputParser.parseToVector(inputVector);
    inputParser.parseToMatrix(inputMatrix, 15, 15);

    /* Parse Weights and Biases */
    Parser weightsParser("weights_and_biases.txt");

    // Parse Weights
    weightsParser.parseWeights(weightsL1, 1, 255, 98);
    weightsParser.parseWeights(weightsL2, 2, 98, 65);
    weightsParser.parseWeights(weightsL3, 3, 65, 50);
    weightsParser.parseWeights(weightsL4, 4, 50, 30);
    weightsParser.parseWeights(weightsL5, 5, 30, 25);
    weightsParser.parseWeights(weightsL6, 6, 25, 40);
    weightsParser.parseWeights(weightsL7, 7, 40, 52);

    // Parse Biases
    weightsParser.parseBiases(biasesL1, 1, 98);
    weightsParser.parseBiases(biasesL2, 2, 65);
    weightsParser.parseBiases(biasesL3, 3, 50);
    weightsParser.parseBiases(biasesL4, 4, 30);
    weightsParser.parseBiases(biasesL5, 5, 25);
    weightsParser.parseBiases(biasesL6, 6, 40);
    weightsParser.parseBiases(biasesL7, 7, 52);

    // Output parsed data (for testing purposes)
    if (0) {
        cout << "Parsed Input Vector:" << endl;         // Not working yet
        for (auto num : inputVector) {
            cout << num << " ";
        }
        cout << endl;
    } else if (0) {
        cout << "Parsed Input Matrix:" << endl;         // Not working yet
        for (auto& row : inputMatrix) {
            for (auto& num : row) {
                cout << num << " ";
            }
            cout << endl;
        }
    } else if (0) {
        cout << "Parsed Weights L1:" << endl;           // Not working yet
        for (auto& row : weightsL1) {
            for (auto& num : row) {
                cout << num << " ";
            }
            cout << endl;
        }
    } else if (1) {
        cout << "Parsed Biases L1:" << endl;            // Working
        for (auto& num : biasesL1) {
            cout << num << " ";
        }
        cout << endl;
    }

    return 1;
}
