#ifndef LOAD_H_
#define LOAD_H_

#include <fstream> 
#include <iterator> 

#include "graphic.h"

// Load state
#define LOAD_SUCCESS 0
#define LOAD_FAILURE 1
#define NO_PROCESSED 2
#define STANDBY_LOAD 3
#define LOADING 4
#define ENDED_PROCESS 5

//File path char
#define MAX_FILE_PATH_CHAR 10
class FILELOAD
{
    public :
    int loadStatus = NO_PROCESSED;
    int loadBinary(char file_path[MAX_FILE_PATH_CHAR]);
};

class LOAD_BMP : public FILELOAD
{
    public :
    IMAGE image;
    int readBinary();
};




#endif LOAD_H_