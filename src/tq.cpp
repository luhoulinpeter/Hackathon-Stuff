#include "tq.h"


/**
 * Queue node
 */
struct tq_node {
    void* data;
    tq_node* next;

    // Constructor
    tq_node (void* _data) : data (_data) {
        next = nullptr;
    }
};


/**
 * Queue constructor
 * Initializes an empty queue
 */
tq::tq () {
    std::lock_guard <std::mutex> guard (mtx);
    len = 0;
    first = nullptr;
    last = nullptr;
}


/**
 * Adds an element to the back of a queue
 * Takes void* as a parameter
 */
void tq::push (void* data) {
    std::lock_guard <std::mutex> guard (mtx);
    if (len == 0) {
        last = new tq_node (data);
        first = last;
    }
    else {
        last -> next = new tq_node (data);
        last = last -> next;
    }
    len ++;
}


/**
 * Removes an element from the front of a queue
 */
void tq::pop () {
    std::lock_guard <std::mutex> guard (mtx);
    if (len > 0) {
        tq_node* tmp = first;
        first = first -> next;
        delete tmp;
        len --;
    }
}


/**
 * Get first element in a queue
 */
void* tq::front () {
    std::lock_guard <std::mutex> guard (mtx);
    return len == 0 ? nullptr : first -> data;
}


/**
 * Get number of elements in a queue
 */
int tq::size () {
    std::lock_guard <std::mutex> guard (mtx);
    return len;
}


/**
 * Check if queue is empty
 */
bool tq::empty () {
    std::lock_guard <std::mutex> guard (mtx);
    return len == 0;
}