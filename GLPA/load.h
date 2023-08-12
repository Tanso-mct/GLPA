#ifndef LOAD_H_
#define LOAD_H_

#include "graphic.h"

// Load state
#define LOAD_SUCCESS 0
#define LOAD_FAILURE 1
#define NO_PROCESSED 2
#define STANDBY_LOAD 3
#define LOADING 4
#define ENDED_PROCESS 5

class FILELOAD
{
    public :
    int loadStatus = NO_PROCESSED;
    char *loadBinary();
};

class LOAD_BMP : public FILELOAD
{
    public :
    IMAGE image;
    int readBinary();
};




#endif LOAD_H_