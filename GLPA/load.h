#ifndef LOAD_H_
#define LOAD_H_

#include <string>
#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#define _CRT_SECURE_NO_WARNINGS

using namespace std;

#include <vector>

#include "graphic.h"

// Load state
#define LOAD_SUCCESS 0
#define LOAD_FAILURE 1
#define NO_PROCESSED 2
#define STANDBY_LOAD 3
#define LOADING 4
#define ENDED_PROCESS 5

//File path char
#define MAX_FILE_PATH_CHAR 11

class FILELOAD
{
    public :
    int loadStatus = NO_PROCESSED;
    int loadBinary(string file_type, string file_name);
};

class LOAD_BMP : public FILELOAD
{
    public :
    // IMAGE image;
    // int readBinary();
};

extern LOAD_BMP sampleBmpFile;




#endif LOAD_H_