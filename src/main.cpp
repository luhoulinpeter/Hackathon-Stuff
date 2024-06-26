/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * StartHack Hackathon HPC Neural Network on Digit Recognition Sponsered by QDX  *
 *                                   Authors                                     *
 *                      Carlvince, Lucas, Peter, Volodmyr                        *
 *                                                                               *
 *                    (SHORT DESCRIPTION OF NEURAL NETWORK)                      *
 *                                                                               *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "reader.h"
#include "model.h"
#include <iostream>
#include <vector>
#include <map>
#include <filesystem>

using namespace std;

// Temporary function to convert vector <long double> to double*
double* v_to_a (const vector <long double>& vec) {
    double* arr = new double [vec.size ()];
    for (size_t i = 0; i < vec.size (); i ++) {
        arr [i] = vec [i];
    }
    return arr;
}

// Temporary function to convert vector < vector <long double> > to double*
double* v2D_to_a (const vector <vector <long double> >& weights) {
    int lsize = weights [0].size ();
    double* arr = new double [weights.size () * lsize];
    for (size_t i = 0; i < weights.size (); i ++) {
        for (int j = 0; j < lsize; j ++) {
            arr [i*lsize + j] = weights [i] [j];
        }
    }
    return arr;
}

int main () {
    // Look-Up Dictionary
    map<int, char> letterMap;
    for (int i = 0; i < 52; ++i) {
        if (i % 2 == 1) {
            letterMap[i] = 'A' + (i) / 2;  
        } else {
            letterMap[i] = 'a' + (i - 1) / 2;  
        }
    }

    // Create matrices and vectors
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

    // Parse Input Tensor
    Parser tensorParser("tensors/01out.txt");

    // Parse to Vector and Matrix
    tensorParser.parseToVector(inputVector);
    tensorParser.parseToMatrix(inputMatrix, 15, 15);

    // Parse Weights and Biases
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
    // What a nice temporary solution!
    model.add_layer (98, v2D_to_a (weightsL1), v_to_a (biasesL1));
    model.add_layer (65, v2D_to_a (weightsL2), v_to_a (biasesL2));
    model.add_layer (50, v2D_to_a (weightsL3), v_to_a (biasesL3));
    model.add_layer (30, v2D_to_a (weightsL4), v_to_a (biasesL4));
    model.add_layer (25, v2D_to_a (weightsL5), v_to_a (biasesL5));
    model.add_layer (40, v2D_to_a (weightsL6), v_to_a (biasesL6));
    model.add_layer (52, v2D_to_a (weightsL7), v_to_a (biasesL7));
    
    ofstream file("results.csv");
    file << "image_number, guess" << endl;
    string tensors_path = filesystem::current_path ().string () + "/tensors";
    int image_number = 1;
    for (auto& entry : filesystem::directory_iterator (tensors_path)) {
        vector <long double> input;
        Parser tensorParser (entry.path ().string ());
        tensorParser.parseToVector (input);
        int res = model.forward_pass (v_to_a (input));
        cout << "Result: " << (res % 2 ? char (97 + res / 2) : char (65 + res / 2)) << endl;
        file << image_number << ", " << letterMap[model.forward_pass (v_to_a (input))] << endl;
        image_number++;
    }
    file.close ();

    return 0;
}