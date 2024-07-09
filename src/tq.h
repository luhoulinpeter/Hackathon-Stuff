#ifndef TQ_H
#define TQ_H

#include <mutex>
#include <string>

// Queue node
struct tq_node;

// Thread safe bare minimum queue structure
struct tq {
private:
    int len;
    tq_node* first;
    tq_node* last;
    std::mutex mtx;

public:
    tq ();
    void push (void* data);
    void pop ();
    void* front ();
    int size ();
    bool empty ();
};

#endif // TQ_H