#ifndef PNG_H_
#define PNG_H_

#include <Windows.h>
#include <string>

// #include <FreeImage.h>

class Png
{
public :
    LPDWORD load(const char* filename);


private :
    int width;
    int height;

    std::string name;
    std::string path;
    
    LPDWORD data;

};

#endif PNG_H_
