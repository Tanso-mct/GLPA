#ifndef GLPA_PNG_H_
#define GLPA_PNG_H_

#include "File.h"

#include <Windows.h>
#include <string>
#include <tchar.h>
#include <stdexcept>

namespace Glpa
{

class Png : public Glpa::File
{
protected :
    // int width = 0;
    // int height = 0;
    // int channels = 0;

    // LPDWORD data;

public :
    Png();
    ~Png() override;

    int getWidth() const {return width;}
    void setWidth(int val) {width = val;}

    int getHeight() const {return height;}
    void setHeight(int val) {height = val;}

    LPDWORD getData() const {return data;}

    void load() override;
    void release() override;
};

}

#endif GLPA_PNG_H_

