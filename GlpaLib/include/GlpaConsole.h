#ifndef GLPA_DEBUG_H_
#define GLPA_DEBUG_H_

#include <initializer_list>

#include "GlpaLib.h"
#include "GlpaBase.h"
#include "GlpaEvent.h"

#include "ConsoleScene.h"

namespace Glpa
{

class CmdHelp : public Glpa::Event
{
public :
    CmdHelp() : Glpa::Event("help"){};
    void onEvent() override;
};

class CmdLog : public Glpa::Event
{
public :
    CmdLog() : Glpa::Event("log"){};
    void onEvent() override;
};

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

    static void Log(std::string str);
    static void CmdOutput(std::string str);

    static void Log(const char* file, int line, std::initializer_list<std::string> linesStr);
    static void Log(std::initializer_list<std::string> linesStr);
    static void CmdOutput(std::initializer_list<std::string> linesStr);

    static void AddEvent(Glpa::Event* event);
};

}


#endif GLPA_DEBUG_H_