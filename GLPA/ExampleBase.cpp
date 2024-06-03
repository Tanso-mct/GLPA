#include "ExampleBase.h"

void ExampleBase::setup()
{
    ExampleScene2d* ptExample2d = new ExampleScene2d();

    addScene(ptExample2d);
}