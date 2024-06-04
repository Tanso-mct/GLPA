#ifndef EXAMPLE_BASE_H_
#define EXAMPLE_BASE_H_

#include "GlpaBase.h"

#include "ExampleScene2d.h"

class ExampleBase : public GlpaBase
{
private :
    ExampleScene2d* ptExample2d;

public :
    ~ExampleBase() override;
    void setup() override;
};

#endif EXAMPLE_BASE_H_
