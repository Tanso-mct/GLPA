#include "GlpaConsole.h"

std::string Glpa::Console::consoleText = "";

Glpa::Console::Console()
{
}

Glpa::Console::~Console()
{
}

void Glpa::Console::setup()
{
    SetBgColor(Glpa::COLOR_BLACK);
    pTexts = new Glpa::Text();
    consoleText = "Graphic Loop Painter [Version 1.0.0]\n";
    consoleText += "Type 'help' for more information.\n";
    consoleText += "\n";
    consoleText += ">";
    
    pTexts->EditWords(consoleText);
    pTexts->EditFontName("Cascadia Mono");
    pTexts->EditFontSize(18);
    pTexts->EditBaselineOffSet(20);
    pTexts->EditLineSpacing(20);
    pTexts->EditPos(Glpa::Vec2d(10, 5));
    pTexts->EditSize(Glpa::Vec2d(GetWindowWidth(), GetWindowHeight()));
    AddSceneObject(pTexts);
}

void Glpa::Console::start()
{
}

void Glpa::Console::update()
{
}