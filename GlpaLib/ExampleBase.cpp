#include "ExampleBase.h"
#include "GlpaConsole.h"

ExampleBaseA::~ExampleBaseA()
{
    Glpa::Console::Log(Glpa::CONSOLE_TAG_EXAMPLE, __FILE__, __LINE__, {"ExampleBaseA Destructor\n"});
}

ExampleBaseA::ExampleBaseA()
{
    Glpa::Console::Log(Glpa::CONSOLE_TAG_EXAMPLE, __FILE__, __LINE__, {"ExampleBaseA Constructor\n"});
}

void ExampleBaseA::setup()
{
    ptExample2d = new ExampleScene2d();
    ptExample2d->setName("example 2d");
    
    AddScene(ptExample2d);
    SetFirstSc(ptExample2d);
}

ExampleBaseB::ExampleBaseB()
{
    Glpa::Console::Log(Glpa::CONSOLE_TAG_EXAMPLE, __FILE__, __LINE__, {"ExampleBaseB Constructor\n"});
}

ExampleBaseB::~ExampleBaseB()
{
    Glpa::Console::Log(Glpa::CONSOLE_TAG_EXAMPLE, __FILE__, __LINE__, {"ExampleBaseB Destructor\n"});
}

void ExampleBaseB::setup()
{
    ptExample3d = new ExampleScene3d();
    ptExample3d->setName("example 2d");
    
    AddScene(ptExample3d);
    SetFirstSc(ptExample3d);
}
