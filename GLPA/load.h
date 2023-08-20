#ifndef LOAD_H_
#define LOAD_H_

#include <string>
#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#define _CRT_SECURE_NO_WARNINGS

#include <vector>

#include "graphic.h"

#define FILETYPE_BMP 0
#define FILETYPE_PNG 1

// Load state
#define LOAD_SUCCESS 0
#define LOAD_FAILURE 1
#define NO_PROCESSED 2
#define STANDBY_LOAD 3
#define LOADING 4
#define ENDED_PROCESS 5

typedef struct tagRGBA
{
    int r;
    int g;
    int b;
    int a;
} RGBA;

typedef struct tagIMAGE
{
    int width;
    int height;
    int colorDepth;
    int compType;
    int format;
} IMAGE;

class FILELOAD
{
    public :
    std::string fileName;
    std::vector<std::string> binaryData;
    int loadStatus = NO_PROCESSED;
    int loadBinary(int file_type, std::string input_file_name);
    void checkBinary();
};

class BMP_FILE : public FILELOAD
{
    public :
    IMAGE image;
    int readBinary();
};

extern BMP_FILE sampleBmpFile;




#endif LOAD_H_