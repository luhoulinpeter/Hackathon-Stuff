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
#include "input.h"

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

    // free used resources (if any)

    return 1;
}
