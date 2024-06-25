/*
 * Functions that parse input array to either Vector or Matrix
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

using namespace std;

class Parser {
    private:
        string FILE_NAME;

        

    public:
        /*
         * Constructor
         */
        Parser(string fileName) {
            FILE_NAME = fileName;
        }

        /*
         * Function: Parses values from file to a vector
         */
        void ParseToVector(long double *vector) {
            ifstream file(FILE_NAME);
            
            string line;
            getline(file, line); 

            stringstream stream(line);
            string token;
            int count {};

            // Parse numbers into array
            while (getline(stream, token, ',') && count < 255) {
                vector[count++] = stoi(token);
            }
        }

        /*
         * Function: Parses values from file to a matrix
         */
        void ParseToMatrix(long double **matrix) {
            ifstream file(FILE_NAME);
            
            string line;
            getline(file, line); 

            stringstream stream(line);
            string token;
            int count {};

            // Parse numbers into array
            while (getline(stream, token, ',') && count < 255) {
                matrix[count % 15][count / 15] = stoi(token);
                count++;
            }
        }

        /*
         * Function: Parses Weights from a file to a list of matrices
         */
        void ParseWeights(long double ***weights, int size) {
            ifstream file(FILE_NAME);

            //string line;
            //while (getline(file, line)) {
            //    if (regex_match(line, regex(R"(.*weights:$)"))) {
            //        getline(file,line);
            //
            //        stringstream stream(line);
            //        string token;
            //        int count {};
            //
            //    }
            //
            //}

        }

        /*
         * Function: Parses Biases from a file to a list of vectors
         */
        void ParseBiases(long double **biases, int size) {

        }
        
        
    
};