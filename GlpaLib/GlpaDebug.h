#ifndef GLPA_DEBUG_H_
#define GLPA_DEBUG_H_

#include "GlpaLib.h"
#include "GlpaBase.h"

#include "GlpaConsole.h"

namespace Glpa
{

class Debug : public GlpaBase
{
private :
    static Debug* instance;
    Debug();

public :
    ~Debug() override;
    void setup() override;

    static void CreateDebugConsole();
};

}


#endif GLPA_DEBUG_H_