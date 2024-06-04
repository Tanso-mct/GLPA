#include "ExampleBase.h"

ExampleBase::~ExampleBase()
{
}

void ExampleBase::setup()
{
    ptExample2d = new ExampleScene2d();
    AddScene(ptExample2d);

    setFirstSc(ptExample2d);
}