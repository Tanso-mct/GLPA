#ifndef PNG_H_
#define PNG_H_

#include <Windows.h>
#include <string>

#include <tchar.h>


class Png
{
public :
    bool load(std::string file_path);

    int width = 0;
    int height = 0;
    int channels = 0;

    std::string name;
    std::string path;
    
    LPDWORD data;

};

#endif PNG_H_
