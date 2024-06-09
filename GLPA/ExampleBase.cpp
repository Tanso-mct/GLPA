#include "ExampleBase.h"
#include "ExampleScene2d.h"

void ExampleScene2d::openExample3d()
{
    ExampleBaseB* pBcB = new ExampleBaseB();
    pBcB->window->setName(L"Example Base B");
    pBcB->window->setApiClassName(L"example_base_b");

    pBcB->window->setWidth(1920);
    pBcB->window->setHeight(1080);

    GlpaLib::AddBase(pBcB);

    GlpaLib::CreateWindowNotApi(pBcB);
    GlpaLib::ShowWindowNotApi(pBcB, SW_SHOWDEFAULT);

    GlpaLib::Load(pBcB);
}

ExampleBaseA::~ExampleBaseA()
{
}

void ExampleBaseA::setup()
{
    ptExample2d = new ExampleScene2d();
    ptExample2d->setName("example 2d");
    
    AddScene(ptExample2d);

    SetFirstSc(ptExample2d);
}

ExampleBaseB::~ExampleBaseB()
{
}

void ExampleBaseB::setup()
{
    ptExample3d = new ExampleScene3d();
    ptExample3d->setName("example 2d");
    
    AddScene(ptExample3d);

    SetFirstSc(ptExample3d);
}
