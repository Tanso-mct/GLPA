#ifndef GLPA_DEBUG_H_
#define GLPA_DEBUG_H_

#include "GlpaLib.h"
#include "GlpaBase.h"

#include "GlpaConsole.h"

namespace Glpa
{

class Console : public GlpaBase
{
private :
    std::wstring_convert<std::codecvt_utf8<wchar_t>> strConverter;

    std::string windowName = "Glpa Console";
    std::string windowApiName = "glpa_console";
    float windowWidth = 1200;
    float windowHeight = 800;

    static Console* instance;
    static Glpa::ConsoleScene* ptConsole;

    Console();

public :
    ~Console() override;
    void setup() override;

    static void Create();
};

}


#endif GLPA_DEBUG_H_