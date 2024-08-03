#ifndef GLPA_CONSOLE_H_
#define GLPA_CONSOLE_H_

#include "Scene2d.h"
#include "Text.h"

namespace Glpa
{

class Console : public Glpa::Scene2d
{
private :
    Glpa::Text* pTexts;
public :
    static std::string consoleText;

    Console();
    ~Console() override;

    void setup() override;
    void start() override;
    void update() override;
};

}


#endif GLPA_CONSOLE_H_

