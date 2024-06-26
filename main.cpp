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
    vector<long double> inputVector;
    vector<vector<long double>> inputMatrix;

    vector<vector<long double>> weightsL1;
    vector<vector<long double>> weightsL2;
    vector<vector<long double>> weightsL3;
    vector<vector<long double>> weightsL4;
    vector<vector<long double>> weightsL5;
    vector<vector<long double>> weightsL6;
    vector<vector<long double>> weightsL7;

    vector<long double> biasesL1;
    vector<long double> biasesL2;
    vector<long double> biasesL3;
    vector<long double> biasesL4;
    vector<long double> biasesL5;
    vector<long double> biasesL6;
    vector<long double> biasesL7;

    /* Parse Template Values */
    Parser parser("weights_and_biases.txt");

    // Parse Inputs
    parser.parseToVector(inputVector);
    parser.parseToMatrix(inputMatrix, 15, 15);

    // Parse Weights
    parser.parseWeights(weightsL1, 1, 255, 98);
    parser.parseWeights(weightsL2, 2, 98, 65);
    parser.parseWeights(weightsL3, 3, 65, 50);
    parser.parseWeights(weightsL4, 4, 50, 30);
    parser.parseWeights(weightsL5, 5, 30, 25);
    parser.parseWeights(weightsL6, 6, 25, 40);
    parser.parseWeights(weightsL7, 7, 40, 52);

    // Parse Biases
    parser.parseBiases(biasesL1, 1, 98);
    parser.parseBiases(biasesL2, 2, 65);
    parser.parseBiases(biasesL3, 3, 50);
    parser.parseBiases(biasesL4, 4, 30);
    parser.parseBiases(biasesL5, 5, 25);
    parser.parseBiases(biasesL6, 6, 40);
    parser.parseBiases(biasesL7, 7, 52);


    return 1;
}
