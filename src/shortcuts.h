#ifndef SHORTCUTS_H
#define SHORTCUTS_H

#include <chrono>
#include <thread>

using namespace std;

#define NOW start = chrono::high_resolution_clock::now ()
#define ELAPSED chrono::duration_cast <chrono::microseconds> (chrono::high_resolution_clock::now () - start).count () / 1000.0
//#define IDLE this_thread::sleep_for (chrono::nanoseconds (10));
#define IDLE this_thread::sleep_for (chrono::nanoseconds (1));
//#define IDLE ;

#endif