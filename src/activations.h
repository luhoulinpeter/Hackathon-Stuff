#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <vector>

using namespace std;
// softmax declaration
vector<long double> softmax(vector<long double> input, int length);

// rectified linear declaration
vector<long double> reLU(vector<long double> input);

#endif