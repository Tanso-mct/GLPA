#ifndef GLPA_DEBUG_H_
#define GLPA_DEBUG_H_

#include <initializer_list>

#include "GlpaLib.h"
#include "GlpaBase.h"
#include "GlpaEvent.h"

#include "ConsoleScene.h"
#include "ExampleBase.h"

namespace Glpa
{

constexpr bool CONSOLE_LOG = true;
constexpr bool CONSOLE_TAG_FILTER = true;

constexpr const char* CONSOLE_TAG_GLPA_LIB = "tag_glpa_lib";
constexpr const char* CONSOLE_TAG_GLPA_RENDER = "tag_glpa_render";
constexpr const char* CONSOLE_TAG_CONSOLE = "tag_console";
constexpr const char* CONSOLE_TAG_EXAMPLE = "tag_example";

constexpr const char* CONSOLE_TAG_ENABLE_FILTERS[] 
= {
    CONSOLE_TAG_GLPA_LIB,
    CONSOLE_TAG_GLPA_RENDER
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

    static void Log(std::string tag, std::string str);
    static void Log(std::string tag, const char* file, int line, std::initializer_list<std::string> linesStr);
    static void Log(std::string tag, std::initializer_list<std::string> linesStr);

    static void CmdOutput(std::string str);
    static void CmdOutput(std::initializer_list<std::string> linesStr);

private :
    class CmdBase : public Glpa::EventList
    {
    private :
        ExampleBaseA* baseA = nullptr;
        ExampleBaseB* baseB = nullptr;

    public :
        CmdBase();
        ~CmdBase() override;

    private :
        class CmdCreate : public Glpa::Event
        {
        private :
            bool baseACreated = false;
            ExampleBaseA* ptBaseA = nullptr;

            bool baseBCreated = false;
            ExampleBaseB* ptBaseB = nullptr;

            enum class eArgs
            {
                type, // string
            };

            std::vector<std::string> typeCds;

        public :
            CmdCreate(ExampleBaseA* argBaseA, ExampleBaseB* argBaseB);
            ~CmdCreate() override;
            bool onEvent(std::vector<std::string> args) override;

            void createBaseA();
            void createBaseB();
        };
    };

};

}


#endif GLPA_DEBUG_H_