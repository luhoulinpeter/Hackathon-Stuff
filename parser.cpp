/*
 * Functions that parse input array to either Vector or Matrix
 */

#include <fstream>
#include <iostream>
#include <string>

using namespace std;

class Parser {
    private:
        string fileName;

    public:
        /*
         * Constructor
         */
        Parser(string fileName) {
            this->fileName = fileName;
        }

        /*
         * Function: Parses values from file to a vector
         */
        void ParseToVector(int (&array)[225]) {
            
        }

        /*
         * Function: Parses values from file to a matrix
         */
        void ParseToMatrix(int (&array)[15][15]) {

        }
        
        
    
};