#ifndef GLPA_DEBUG_H_
#define GLPA_DEBUG_H_

#include "GlpaLib.h"
#include "GlpaBase.h"

#include "GlpaConsole.h"

namespace Glpa
{

class Debug : public GlpaBase
{
public :
    Debug();
    ~Debug() override;
    void setup() override;
};

}


#endif GLPA_DEBUG_H_