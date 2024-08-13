#include "ConsoleScene.h"
#include "GlpaLog.h"

Glpa::ConsoleScene::ConsoleScene()
{
}

Glpa::ConsoleScene::~ConsoleScene()
{
    
}

void Glpa::ConsoleScene::setup()
{
    CmdText* pCmdText = new CmdText(this);
    Glpa::EventManager::AddEventList(pCmdText);

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
    pCommandText->EditPos(cmdTextBasePos);
    pCommandText->EditSize(Glpa::Vec2d(GetWindowWidth() / 2, GetWindowHeight()));
    AddSceneObject(pCommandText);

    pLogText = new Glpa::Text("log_text");
    logTextSize = logText.size();

    logTextBasePos = Glpa::Vec2d(GetWindowWidth() / 2 + cmdTextBasePos.x, cmdTextBasePos.y);

    pLogText->EditWords(logText);
    pLogText->EditFontName("Cascadia Mono");
    pLogText->EditFontSize(fontSize);
    pLogText->EditBaselineOffSet(baseLineOffSet);
    pLogText->EditLineSpacing(lineSpacing);
    pLogText->EditPos(Glpa::Vec2d(logTextBasePos.x, logTextBasePos.y));
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

    scrollWindow();
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
                pCommandText->EditWords(commandText);
            }

            std::string strCommand 
            = commandText.substr(commandTextSize, commandText.size() - commandTextSize);

            commandText += "\n";
            if (!Glpa::EventManager::ExecuteEvent(strCommand))
            {
                writeCmdLog
                ({
                    "'" + strCommand + "' is not recognized as an command",
                    "A list of executable commands can be found with 'help'."
                });
            }

            commandText += "\n>";
            pCommandText->EditWords(commandText);
            commandTextSize = commandText.size();
            isTypingAnimRestart = true;

            autoScrollCmdText();
            autoScrollLogText();
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

void Glpa::ConsoleScene::autoScrollCmdText()
{
    int lineCount = pCommandText->GetLineCount() + 2 + margin;
    int couldLineCount = GetWindowHeight() / lineSpacing;

    int scrollCount = lineCount - couldLineCount;

    if (scrollCount > 0)
    {
        pCommandText->EditPos(Glpa::Vec2d(cmdTextBasePos.x, cmdTextBasePos.y - lineSpacing * scrollCount));
    }
    else
    {
        pCommandText->EditPos(cmdTextBasePos);
    }
}

void Glpa::ConsoleScene::autoScrollLogText()
{
    int lineCount = pLogText->GetLineCount() + 2 + margin;
    int couldLineCount = GetWindowHeight() / lineSpacing;

    int scrollCount = lineCount - couldLineCount;

    if (scrollCount > 0)
    {
        pLogText->EditPos(Glpa::Vec2d(logTextBasePos.x, logTextBasePos.y - lineSpacing * scrollCount));
    }
    else
    {
        pLogText->EditPos(Glpa::Vec2d(logTextBasePos.x, logTextBasePos.y));
    }
}

void Glpa::ConsoleScene::scrollWindow()
{
    if (window->isFocusing)
    {
        int moveAmount = 0;
        if (GetNowMouseMsg(Glpa::CHAR_MOUSE_WHEEL, moveAmount))
        {
            moveAmount /= Glpa::INT_MOUSE_WHEEL_MOVE * -1;
            Glpa::Vec2d nowMousePos = GetNowMousePos();
            if (nowMousePos.x < logTextBasePos.x)
            {
                if (GetNowKeyMsg(Glpa::CHAR_LALT)) moveAmount *= scrollWithAltSpeed;
                int scrollCmdCount = moveAmount;

                int lineCount = pCommandText->GetLineCount() + 2 + margin;
                int couldLineCount = GetWindowHeight() / lineSpacing;
                if (lineCount >= scrollCmdCount + couldLineCount)
                {
                    Glpa::Vec2d pos = pCommandText->GetPos();
                    float maxY = cmdTextBasePos.y - lineSpacing * (lineCount - couldLineCount);

                    if 
                    (
                        pos.y - lineSpacing * scrollCmdCount >= maxY && 
                        pos.y - lineSpacing * scrollCmdCount <= cmdTextBasePos.y
                    ){
                        pCommandText->EditPos(Glpa::Vec2d(pos.x, pos.y - lineSpacing * scrollCmdCount));
                    }
                    else if (pos.y - lineSpacing * scrollCmdCount < maxY)
                    {
                        pCommandText->EditPos(Glpa::Vec2d(pos.x, maxY));
                    }
                    else if (pos.y - lineSpacing * scrollCmdCount > cmdTextBasePos.y)
                    {
                        pCommandText->EditPos(cmdTextBasePos);
                    }
                }
            }
            else
            {
                if (GetNowKeyMsg(Glpa::CHAR_LALT)) moveAmount *= scrollWithAltSpeed;
                int scrollLogCount = moveAmount;

                int lineCount = pLogText->GetLineCount() + 2 + margin;
                int couldLineCount = GetWindowHeight() / lineSpacing;
                if (lineCount >= scrollLogCount + couldLineCount)
                {
                    Glpa::Vec2d pos = pLogText->GetPos();
                    float maxY = logTextBasePos.y - lineSpacing * (lineCount - couldLineCount);

                    if 
                    (
                        pos.y - lineSpacing * scrollLogCount >= maxY && 
                        pos.y - lineSpacing * scrollLogCount <= logTextBasePos.y
                    ){
                        pLogText->EditPos(Glpa::Vec2d(pos.x, pos.y - lineSpacing * scrollLogCount));
                    }
                    else if (pos.y - lineSpacing * scrollLogCount < maxY)
                    {
                        pLogText->EditPos(Glpa::Vec2d(pos.x, maxY));
                    }
                    else if (pos.y - lineSpacing * scrollLogCount > logTextBasePos.y)
                    {
                        pLogText->EditPos(logTextBasePos);
                    }
                }
            }
        }
    }
}

void Glpa::ConsoleScene::writeLog(std::initializer_list<std::string> strLines, bool isLastNewLine)
{
    for (std::string line : strLines)
    {
        logText += line + "\n";
    }

    if (!isLastNewLine)
    {
        logText.pop_back();
    }

    pLogText->EditWords(logText);
    logTextSize = logText.size();

    autoScrollLogText();
}

void Glpa::ConsoleScene::writeCmdLog(std::initializer_list<std::string> strLines)
{
    for (std::string line : strLines)
    {
        commandText += line + "\n";
    }
    pCommandText->EditWords(commandText);
    commandTextSize = commandText.size();

    autoScrollCmdText();
}


Glpa::ConsoleScene::CmdText::CmdCount::CmdCount(Glpa::ConsoleScene *argBase)
: Glpa::Event("count", __FILE__, __LINE__, {{"line", "word"}, {"cmd", "log"}, {}})
{
    base = argBase;

    typeCds.push_back("line");
    typeCds.push_back("word");

    textCds.push_back("cmd");
    textCds.push_back("log");
}

Glpa::ConsoleScene::CmdText::CmdCount::~CmdCount()
{
}

bool Glpa::ConsoleScene::CmdText::CmdCount::onEvent(std::vector<std::string> args)
{
    std::string thisType = args[static_cast<int>(eArgs::type)];
    std::string thisText = args[static_cast<int>(eArgs::text)];

    int line = -1;
    if (args.size() == 3)
    {
        line = std::stoi(args[static_cast<int>(eArgs::line)]);
    }

    if (thisType == typeCds[0])
    {
        GetLineCount(thisText);
        return true;
    }
    else if (thisType == typeCds[1] && line != -1)
    {
        GetWordCount(thisText, line);
        return true;
    }
    else
    {
        return false;
    }

}

void Glpa::ConsoleScene::CmdText::CmdCount::GetLineCount(std::string thisText)
{
    if (thisText == textCds[0])
    {
        int lineCount = base->pCommandText->GetLineCount();
        int lastLineWordsCount = base->pCommandText->GetLineTextCount(lineCount - 1);
        base->writeCmdLog
        ({
            "Cmd line count: " + std::to_string(lineCount),
            "Cmd last line words count: " + std::to_string(lastLineWordsCount)
        });
    }
    else if (thisText == textCds[1])
    {
        int lineCount = base->pLogText->GetLineCount();
        int lastLineWordsCount = base->pLogText->GetLineTextCount(lineCount - 1);
        base->writeCmdLog
        ({
            "Log line count: " + std::to_string(lineCount),
            "Log last line words count: " + std::to_string(lastLineWordsCount)
        });
    }
}

void Glpa::ConsoleScene::CmdText::CmdCount::GetWordCount(std::string thisText, int line)
{
    if (thisText == textCds[0])
    {
        int wordsCount = base->pCommandText->GetLineTextCount(line);
        base->writeCmdLog
        ({
            "Cmd line " + std::to_string(line) + " words count : " + std::to_string(wordsCount)
        });
    }
    else if (thisText == textCds[1])
    {
        int wordsCount = base->pLogText->GetLineTextCount(line);
        base->writeCmdLog
        ({
            "Log line " + std::to_string(line) + " words count : " + std::to_string(wordsCount)
        });
    }
}

Glpa::ConsoleScene::CmdText::CmdText(Glpa::ConsoleScene *argBase)  : Glpa::EventList("text")
{
    AddEvent(new CmdCount(argBase));
}

Glpa::ConsoleScene::CmdText::~CmdText()
{

}
