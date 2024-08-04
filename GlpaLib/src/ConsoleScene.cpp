#include "ConsoleScene.h"

Glpa::ConsoleScene::ConsoleScene()
{
}

Glpa::ConsoleScene::~ConsoleScene()
{
    for (auto it = events.begin(); it != events.end(); it++)
    {
        delete it->second;
    }
}

void Glpa::ConsoleScene::setup()
{
    SetBgColor(Glpa::COLOR_BLACK);

    std::string folderPath = "resource/Assets/Images/";
    pBackground = new Glpa::Image("debug_console_background",folderPath + "debug_console_background.png", Glpa::Vec2d(0, 0));
    AddSceneObject(pBackground);

    pCommandText = new Glpa::Text("command_text");
    commandTextSize = commandText.size();
    
    pCommandText->EditWords(commandText);
    pCommandText->EditFontName("Cascadia Mono");
    pCommandText->EditFontSize(fontSize);
    pCommandText->EditBaselineOffSet(baseLineOffSet);
    pCommandText->EditLineSpacing(lineSpacing);
    pCommandText->EditPos(textBasePos);
    pCommandText->EditSize(Glpa::Vec2d(GetWindowWidth() / 2, GetWindowHeight()));
    AddSceneObject(pCommandText);

    pLogText = new Glpa::Text("log_text");
    logTextSize = logText.size();

    pLogText->EditWords(logText);
    pLogText->EditFontName("Cascadia Mono");
    pLogText->EditFontSize(fontSize);
    pLogText->EditBaselineOffSet(baseLineOffSet);
    pLogText->EditLineSpacing(lineSpacing);
    pLogText->EditPos(Glpa::Vec2d(GetWindowWidth() / 2 + textBasePos.x, textBasePos.y));
    pLogText->EditSize(Glpa::Vec2d(GetWindowWidth(), GetWindowHeight()));
    AddSceneObject(pLogText);
}

void Glpa::ConsoleScene::start()
{
    startTime = std::chrono::high_resolution_clock::now();
    lastTime = std::chrono::high_resolution_clock::now();
}

void Glpa::ConsoleScene::update()
{
    currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> delta = std::chrono::duration_cast<std::chrono::duration<float>>(currentTime - lastTime);
    deltaTime = delta.count();

    if (window->isFocusing) isTyping = true;
    else isTyping = false;

    typeWord();
    if (isTyping) typeAnim();
    else
    {
        if (commandText.back() == '|')
        {
            isTypingAnimRestart = true;
            commandText.pop_back();
            pCommandText->EditWords(commandText);
        }
    }

    lastTime = currentTime;
}

void Glpa::ConsoleScene::typeWord()
{
    std::string key = GetNowKeyDownMsg();
    if (!key.empty())
    {
        if (key == Glpa::CHAR_BACKSPACE)
        {
            if (commandText.back() == '|')
            {
                commandText.pop_back();
            }

            if (commandText.size() > commandTextSize)
            {
                commandText.pop_back();
                pCommandText->EditWords(commandText);
            }
            isTypingAnimRestart = true;
        }
        else if (key == Glpa::CHAR_ENTER)
        {
            if (commandText.back() == '|')
            {
                commandText.pop_back();
            }

            std::string strCommand 
            = commandText.substr(commandTextSize, commandText.size() - commandTextSize);

            commandText += "\n";
            executeCommand(strCommand);

            commandText += "\n>";
            pCommandText->EditWords(commandText);
            commandTextSize = commandText.size();
            isTypingAnimRestart = true;
        }
        else if (key == Glpa::CHAR_SPACE)
        {
            if (commandText.back() == '|')
            {
                commandText.pop_back();
            }

            commandText += " ";
            pCommandText->EditWords(commandText);
            isTypingAnimRestart = true;
        }
        else if (IsWord(key))
        {
            if (commandText.back() == '|')
            {
                commandText.pop_back();
            }

            commandText += key;
            pCommandText->EditWords(commandText);
            isTypingAnimRestart = true;
        }
    }
}

void Glpa::ConsoleScene::typeAnim()
{
    if (isTypingAnimRestart)
    {
        elapsedTime = 0;
        isTypingAnimOn = false;
        isTypingAnimRestart = false;
    }

    if (elapsedTime >= 0 && elapsedTime < 0.5)
    {
        if (!isTypingAnimOn)
        {
            isTypingAnimOn = true;
            commandText += "|";
            pCommandText->EditWords(commandText);
        }
        elapsedTime += deltaTime;
    }
    else if (elapsedTime >= 0.5 && elapsedTime < 1)
    {
        if (isTypingAnimOn)
        {
            isTypingAnimOn = false;
            if (commandText.back() == '|')
            {
                commandText.pop_back();
            }

            pCommandText->EditWords(commandText);
        }
        elapsedTime += deltaTime;
    }
    else if (elapsedTime >= 1)
    {
        elapsedTime = 0;
    }
}

void Glpa::ConsoleScene::writeLog(std::string str)
{
    logText += str;
    pLogText->EditWords(logText);
    logTextSize = logText.size();
}

void Glpa::ConsoleScene::writeCmdLog(std::string str)
{
    commandText += str;
    pCommandText->EditWords(commandText);
    commandTextSize = commandText.size();
}

void Glpa::ConsoleScene::writeLog(std::initializer_list<std::string> strLines)
{
    for (std::string line : strLines)
    {
        logText += line + "\n";
    }
    pLogText->EditWords(logText);
    logTextSize = logText.size();
}

void Glpa::ConsoleScene::writeCmdLog(std::initializer_list<std::string> strLines)
{
    for (std::string line : strLines)
    {
        commandText += line + "\n";
    }
    pCommandText->EditWords(commandText);
    commandTextSize = commandText.size();
}

void Glpa::ConsoleScene::addEvent(Glpa::Event *event)
{
    events[event->getName()] = event;
}

void Glpa::ConsoleScene::executeCommand(std::string str)
{
    for (auto it = events.begin(); it != events.end(); it++)
    {
        if (it->first == str)
        {
            it->second->onEvent();
            return;
        }
    }

    writeCmdLog
    ({
        "'" + str + "' is not recognized as an command",
        "A list of executable commands can be found with 'help'."
    });
}

