#include "ExampleBase.h"

ExampleBase::~ExampleBase()
{
}

void ExampleBase::setup()
{
    ptExample2d = new ExampleScene2d();
    ptExample2d->setName("example 2d");
    
    AddScene(ptExample2d);

    SetFirstSc(ptExample2d);
}