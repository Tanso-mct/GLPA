#include "GlpaDebug.h"

Glpa::Debug* Glpa::Debug::instance = nullptr;

Glpa::Debug::Debug()
{
}

Glpa::Debug::~Debug()
{
}

void Glpa::Debug::setup()
{
    Glpa::Console* ptConsole = new Glpa::Console();
    ptConsole->setName("example 2d");
    
    AddScene(ptConsole);
    SetFirstSc(ptConsole);
}

void Glpa::Debug::CreateDebugConsole()
{
    instance = new Glpa::Debug();
    instance->window->SetName(L"Debug Console");
    instance->window->SetApiClassName(L"debug_console");
    GlpaLib::AddBase(instance);
    GlpaLib::CreateWindowNotApi(instance);
    GlpaLib::ShowWindowNotApi(instance, SW_SHOWDEFAULT);
    GlpaLib::Load(instance);
}
