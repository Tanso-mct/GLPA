#ifndef GLPA_CONSOLE_H_
#define GLPA_CONSOLE_H_

#include "Scene2d.h"
#include "Text.h"
#include "GlpaEvent.h"

#include <chrono>
#include <initializer_list>

#include "GlpaEvent.h"

namespace Glpa
{

class ConsoleScene : public Glpa::Scene2d
{
private :
    class CmdTextLastCharPos : public Glpa::Event
    {
    public :
        CmdTextLastCharPos() : Glpa::Event("text last_char pos", __FILE__, __LINE__){};
        void onEvent() override;
    };
    
    Glpa::Image* pBackground = nullptr;
    Glpa::Text* pCommandText = nullptr;
    Glpa::Text* pLogText = nullptr;

    bool isTyping = false;
    bool isTypingAnimRestart = false;
    bool isTypingAnimOn = false;
    float elapsedTime = 0;

    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point currentTime;
    float deltaTime = 0;
    std::chrono::high_resolution_clock::time_point lastTime;

    float fontSize = 18;
    float baseLineOffSet = 20;
    float lineSpacing = 25;

    Vec2d textBasePos = Vec2d(10, 5);

    std::string commandText 
    = "Graphic Loop Painter [Version 1.0.0]\nType 'help' for more information.\n\n>";
    
    std::string logText 
    = "Program log output field.\nType 'log' for more information.\n\n";

    int commandTextSize = 0;
    int logTextSize = 0;

public :
    ConsoleScene();
    ~ConsoleScene() override;

    void setup() override;
    void start() override;
    void update() override;

    void typeWord();
    void typeAnim();

    void writeLog(std::string str);
    void writeCmdLog(std::string str);

    void writeLog(std::initializer_list<std::string> strLines);
    void writeCmdLog(std::initializer_list<std::string> strLines);
};

}


#endif GLPA_CONSOLE_H_

