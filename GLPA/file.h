#ifndef FILE_H_
#define FILE_H_

#include <string>
#include <iostream>
#include <fstream>

#include <stdio.h>
#define _CRT_SECURE_NO_WARNINGS

#include <vector>

#include "graphic.h"
#include "cgmath.cuh"

#define FILETYPE_BMP 0
#define FILETYPE_PNG 1

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

class OBJ_FILE : public FILELOAD
{
    public :
    RANGE_CUBE range;
    VERTEX v;
    POLYGON poly;
    int loadData(std::string inputFileName);
};

class MTL_FILE : public FILELOAD
{
    public :
    VECTOR3D ka;
    VECTOR3D kd;
    int loadData(std::string inputFileName);
};

// BMP files
extern BMP_FILE sampleBmpFile;

// MTL files
extern MTL_FILE tempMtlFile;




#endif FILE_H_