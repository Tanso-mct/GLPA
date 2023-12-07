#ifndef PNG_H_
#define PNG_H_

#include <Windows.h>
#include <string>


class Png
{
public :


private :
    int width;
    int height;

    std::string name;
    std::string path;
    
    LPDWORD data;

};

#endif PNG_H_
