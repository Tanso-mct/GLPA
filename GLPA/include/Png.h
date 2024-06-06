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
    int width = 0;
    int height = 0;
    int channels = 0;

    LPDWORD data;

public :
    ~Png() override;

    void load() override;
    void release() override;
};

}

#endif GLPA_PNG_H_

