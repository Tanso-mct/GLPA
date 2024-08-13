#ifndef EXAMPLE_BASE_H_
#define EXAMPLE_BASE_H_

#include "GlpaBase.h"

#include "ExampleScene2d.h"
#include "ExampleScene3d.h"

class ExampleBaseA : public GlpaBase
{
private :
    ExampleScene2d* ptExample2d;

public :
    ExampleBaseA();
    ~ExampleBaseA() override;
    void setup() override;
};

class ExampleBaseB : public GlpaBase
{
private :
    ExampleScene3d* ptExample3d;

public :
    ExampleBaseB();
    ~ExampleBaseB() override;
    void setup() override;
};

#endif EXAMPLE_BASE_H_
