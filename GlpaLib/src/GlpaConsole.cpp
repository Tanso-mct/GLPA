#include "GlpaConsole.h"

Glpa::Console* Glpa::Console::instance = nullptr;
Glpa::ConsoleScene* Glpa::Console::ptConsole = nullptr;

Glpa::Console::Console()
{
}

Glpa::Console::~Console()
{
}

void Glpa::Console::setup()
{
    ptConsole = new Glpa::ConsoleScene();
    ptConsole->setName("console_scene");
    
    AddScene(ptConsole);
    SetFirstSc(ptConsole);
}

void Glpa::Console::Create()
{
    instance = new Glpa::Console();
    std::wstring wWindowName = instance->strConverter.from_bytes(instance->windowName);
    std::wstring wWindowApiName = instance->strConverter.from_bytes(instance->windowApiName);
    instance->window->SetName(wWindowName.c_str());
    instance->window->SetApiClassName(wWindowApiName.c_str());
    instance->window->SetWidth(instance->windowWidth);
    instance->window->SetHeight(instance->windowHeight);

    GlpaLib::AddBase(instance);
    GlpaLib::CreateWindowNotApi(instance);
    GlpaLib::ShowWindowNotApi(instance, SW_SHOWDEFAULT);
    GlpaLib::Load(instance);

    instance->AddEvent(new Glpa::CmdHelp());
    instance->AddEvent(new Glpa::CmdLog());
}

void Glpa::Console::Log(std::string str)
{
    if (instance == nullptr) return;
    instance->ptConsole->writeLog(str);
}

void Glpa::Console::CmdOutput(std::string str)
{
    if (instance == nullptr) return;
    instance->ptConsole->writeCmdLog(str);
}

void Glpa::Console::Log(const char *file, int line, std::initializer_list<std::string> linesStr)
{
    if (instance == nullptr) return;
    instance->ptConsole->writeLog("[" + std::string(file) + ":" + std::to_string(line) + "]\n");
    instance->ptConsole->writeLog(linesStr);
    instance->ptConsole->writeLog("\n");
}

void Glpa::Console::Log(std::initializer_list<std::string> linesStr)
{
    if (instance == nullptr) return;
    instance->ptConsole->writeLog(linesStr);
    instance->ptConsole->writeLog("\n");
}

void Glpa::Console::CmdOutput(std::initializer_list<std::string> linesStr)
{
    if (instance == nullptr) return;
    instance->ptConsole->writeCmdLog(linesStr);
}

void Glpa::Console::AddEvent(Glpa::Event *event)
{
    if (instance == nullptr) return;
    instance->ptConsole->addEvent(event);
}

void Glpa::CmdHelp::onEvent()
{
    Glpa::Console::CmdOutput
    ({
        "Available commands:",
        "help - Display this message.",
        "log - Display detailed information about Log.",
    });
}

void Glpa::CmdLog::onEvent()
{
    Glpa::Console::CmdOutput
    ({
        "Log command is not yet implemented.",
        "This command will display detailed information about the program log.",
    });
}
