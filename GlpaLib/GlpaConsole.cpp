#include "GlpaConsole.h"

Glpa::Console::~Console()
{
}

void Glpa::Console::setup()
{
    SetBgColor(Glpa::COLOR_BLACK);
    pTexts = new Glpa::Text();
    pTexts->EditWords("Console Texts");
    AddSceneObject(pTexts);
}

void Glpa::Console::start()
{
    OutputDebugStringA("Console::start\n");
}

void Glpa::Console::update()
{
    OutputDebugStringA("Console::update\n");
}