#include "GlpaDebug.h"

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
}
