/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * StartHack Hackathon HPC Neural Network on Digit Recognition Sponsered by QDX  *
 *                                   Authors                                     *
 *                      Carlvince, Lucas, Peter, Volodmyr                        *
 *                                                                               *
 *                    (SHORT DESCRIPTION OF NEURAL NETWORK)                      *
 *                                                                               *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <vector>
#include <map>
#include "parser.h"
#include "model.hpp"

using namespace std;

int main () {
    /* Look-Up Dictionary */
    map<vector<long double>, char> letterMap;

    // Add elements to the map
    for (int i = 0; i < 52; ++i) {
        vector<long double> key(52, 0.0); 
        key[i] = 1.0; 

        if (i % 2 == 1) {
            letterMap[key] = 'A' + (i - 1) / 2;  
        } else {
            letterMap[key] = 'a' + (i - 2) / 2;  
        }
    }
    
    /* Create matrices and vectors */
    vector<long double> inputVector;
    vector<vector<long double> > inputMatrix;

    vector<vector<long double> > weightsL1;
    vector<vector<long double> > weightsL2;
    vector<vector<long double> > weightsL3;
    vector<vector<long double> > weightsL4;
    vector<vector<long double> > weightsL5;
    vector<vector<long double> > weightsL6;
    vector<vector<long double> > weightsL7;

    vector<long double> biasesL1;
    vector<long double> biasesL2;
    vector<long double> biasesL3;
    vector<long double> biasesL4;
    vector<long double> biasesL5;
    vector<long double> biasesL6;
    vector<long double> biasesL7;

    /* Parse Input Tensor */
    Parser tensorParser("tensors/01out.txt");               // Use "\\tensors\\01out.txt" for Windows

    // Parse to Vector and Matrix
    tensorParser.parseToVector(inputVector);
    tensorParser.parseToMatrix(inputMatrix, 15, 15);

    /* Parse Weights and Biases */
    Parser weightsParser("weights_and_biases.txt");

    // Parse Weights
    weightsParser.parseWeights(weightsL1, 1, 225, 98);
    weightsParser.parseWeights(weightsL2, 2, 98, 65);
    weightsParser.parseWeights(weightsL3, 3, 65, 50);
    weightsParser.parseWeights(weightsL4, 4, 50, 30);
    weightsParser.parseWeights(weightsL5, 5, 30, 25);
    weightsParser.parseWeights(weightsL6, 6, 25, 40);
    weightsParser.parseWeights(weightsL7, 7, 40, 52);

    // Parse Biases
    weightsParser.parseBiases(biasesL1, 1);
    weightsParser.parseBiases(biasesL2, 2);
    weightsParser.parseBiases(biasesL3, 3);
    weightsParser.parseBiases(biasesL4, 4);
    weightsParser.parseBiases(biasesL5, 5);
    weightsParser.parseBiases(biasesL6, 6);
    weightsParser.parseBiases(biasesL7, 7);
    
    // Initialize model
    Model model (7, 225);
    model.add_layer (98, nullptr, nullptr);
    model.add_layer (65, nullptr, nullptr);
    model.add_layer (50, nullptr, nullptr);
    model.add_layer (30, nullptr, nullptr);
    model.add_layer (25, nullptr, nullptr);
    model.add_layer (40, nullptr, nullptr);
    model.add_layer (52, nullptr, nullptr);

    return 1;
}
