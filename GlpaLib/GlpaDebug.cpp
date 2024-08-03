#include "GlpaDebug.h"

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