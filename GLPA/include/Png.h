#ifndef GLPA_PNG_H_
#define GLPA_PNG_H_

#include "File.h"

namespace Glpa
{

class Png : public Glpa::File
{
private :

public :
    ~Png() override;

    void load() override;
    void release() override;
};

}

#endif GLPA_PNG_H_

