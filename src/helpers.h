#ifndef HELPERS_H
#define HELPERS_H

#include <chrono>
#include <thread>

#define NOW start = std::chrono::high_resolution_clock::now ()
#define ELAPSED std::chrono::duration_cast <std::chrono::microseconds> (std::chrono::high_resolution_clock::now () - start).count () / 1000.0
//#define IDLE std::this_thread::sleep_for (std::chrono::nanoseconds (10));
#define IDLE std::this_thread::sleep_for (std::chrono::nanoseconds (1));
//#define IDLE ;

#endif // HELPERS_H