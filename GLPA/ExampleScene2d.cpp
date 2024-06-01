#include "ExampleScene2d.h"

void ExampleScene2d::setup()
{
    objs["A"]->load();
    if (Glpa::File* example = dynamic_cast<Glpa::File*>(objs["A"])) 
    {
    }
}