#ifndef GLPA_CONSOLE_H_
#define GLPA_CONSOLE_H_

#include "Scene2d.h"
#include "Text.h"

#include <chrono>

namespace Glpa
{

class ConsoleScene : public Glpa::Scene2d
{
private :
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

    std::string commandText = "";
    std::string logText = "";

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
};

}


#endif GLPA_CONSOLE_H_

