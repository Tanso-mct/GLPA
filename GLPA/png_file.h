#ifndef PNG_FILE_H_
#define PNG_FILE_H_

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

#endif PNG_FILE_H_
