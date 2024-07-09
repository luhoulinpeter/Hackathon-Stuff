#ifndef READER_H
#define READER_H

#include <string>
#include <atomic>

// Read values from file to an array
void read_input (const std::string& filename, double* input, std::atomic_int* ready, std::atomic_int* free_readers);

// Initialize Model
void init_model (const std::string& wab);

#endif 
