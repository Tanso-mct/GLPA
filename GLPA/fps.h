#ifndef FPS_H_
#define FPS_H_

#include <chrono>
#include <thread>
#include <time.h>

class Fps
{
public :
    void initialize();
    void limit();
    double get();

private :
    bool initialized = false;

    double max;
    double last;

    clock_t lastLoop;
};

#endif  FPS_H_


