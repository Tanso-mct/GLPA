#include "GlpaConsole.h"

Glpa::Console* Glpa::Console::instance = nullptr;
Glpa::ConsoleScene* Glpa::Console::ptConsole = nullptr;

Glpa::Console::Console()
{
}

Glpa::Console::~Console()
{
    instance = nullptr;
}

void Glpa::Console::setup()
{
    ptConsole = new Glpa::ConsoleScene();
    ptConsole->setName("console_scene");

    AddScene(ptConsole);
    SetFirstSc(ptConsole);
}

void Glpa::Console::Create()
{
    instance = new Glpa::Console();
    instance->SetName("Console");
    std::wstring wWindowName = instance->strConverter.from_bytes(instance->windowName);
    std::wstring wWindowApiName = instance->strConverter.from_bytes(instance->windowApiName);
    instance->window->SetName(wWindowName.c_str());
    instance->window->SetApiClassName(wWindowApiName.c_str());
    instance->window->SetWidth(instance->windowWidth);
    instance->window->SetHeight(instance->windowHeight);

    GlpaLib::AddBase(instance);
    GlpaLib::CreateWindowNotApi(instance);
    GlpaLib::ShowWindowNotApi(instance, SW_SHOWDEFAULT);
    GlpaLib::Load(instance);

    CmdBase* pCmdBase = new CmdBase();
    Glpa::EventManager::AddEventList(pCmdBase);
}

void Glpa::Console::Log(std::string tag, std::string str)
{
    if (instance == nullptr) return;

    if (Glpa::CONSOLE_LOG)
    {
        if (Glpa::CONSOLE_TAG_FILTER)
        {
            for (auto& tagFilter : Glpa::CONSOLE_TAG_ENABLE_FILTERS)
            {
                if (tag == tagFilter)
                {
                    instance->ptConsole->writeLog({str}, true);
                    break;
                }
            }
        }
        else
        {
            instance->ptConsole->writeLog({str}, true);
        }
    }
}

void Glpa::Console::CmdOutput(std::string str)
{
    if (instance == nullptr) return;
    instance->ptConsole->writeCmdLog({str});
}

void Glpa::Console::Log(std::string tag, const char *file, int line, std::initializer_list<std::string> linesStr)
{
    if (instance == nullptr) return;

    if (Glpa::CONSOLE_LOG)
    {
        if (Glpa::CONSOLE_TAG_FILTER)
        {
            for (auto& tagFilter : Glpa::CONSOLE_TAG_ENABLE_FILTERS)
            {
                if (tag == tagFilter)
                {
                    instance->ptConsole->writeLog({"[" + std::string(file) + ":" + std::to_string(line) + "]"}, false);
                    instance->ptConsole->writeLog({linesStr}, true);
                    break;
                }
            }
        }
        else
        {
            instance->ptConsole->writeLog({"[" + std::string(file) + ":" + std::to_string(line) + "]"}, false);
            instance->ptConsole->writeLog({linesStr}, true);
        }
    }
}

void Glpa::Console::Log(std::string tag, std::initializer_list<std::string> linesStr)
{
    if (instance == nullptr) return;

    if (Glpa::CONSOLE_LOG)
    {
        if (Glpa::CONSOLE_TAG_FILTER)
        {
            for (auto& tagFilter : Glpa::CONSOLE_TAG_ENABLE_FILTERS)
            {
                if (tag == tagFilter)
                {
                    instance->ptConsole->writeLog({linesStr}, true);
                    break;
                }
            }
        }
        else
        {
            instance->ptConsole->writeLog({linesStr}, true);
        }
    }
}

void Glpa::Console::CmdOutput(std::initializer_list<std::string> linesStr)
{
    if (instance == nullptr) return;
    instance->ptConsole->writeCmdLog(linesStr);
}

// void Glpa::CmdHelp::onEvent()
// {
//     Glpa::Console::CmdOutput
//     ({
//         "Available commands:",
//         "help - Display this message.",
//         "log - Display detailed information about Log.",
//     });
// }

// void Glpa::CmdLog::onEvent()
// {
//     Glpa::Console::CmdOutput
//     ({
//         "Log command is not yet implemented.",
//         "This command will display detailed information about the program log.",
//     });
// }

Glpa::Console::CmdBase::CmdBase() : Glpa::EventList("base")
{
    AddEvent(new CmdCreate(baseA, baseB));
}

Glpa::Console::CmdBase::~CmdBase()
{
}

Glpa::Console::CmdBase::CmdCreate::CmdCreate(ExampleBaseA *argBaseA, ExampleBaseB *argBaseB)
: Glpa::Event("create", __FILE__, __LINE__, {{"base_a", "base_b"}})
{
    ptBaseA = argBaseA;
    ptBaseB = argBaseB;

    typeCds.push_back("base_a");
    typeCds.push_back("base_b");
}

Glpa::Console::CmdBase::CmdCreate::~CmdCreate()
{
}

bool Glpa::Console::CmdBase::CmdCreate::onEvent(std::vector<std::string> args)
{
    std::string thisType = args[static_cast<int>(eArgs::type)];

    if (thisType == typeCds[0]) // base_a
    {
        if (!baseACreated)
        {
            createBaseA();
            Glpa::Console::CmdOutput({"Base A created."});
            baseACreated = true;
        }
        else Glpa::Console::CmdOutput({"Base A already created."});

        return true;
    }
    else if (thisType == typeCds[1]) // base_b
    {
        if (!baseBCreated)
        {
            createBaseB();
            Glpa::Console::CmdOutput({"Base B created."});
        }
        else Glpa::Console::CmdOutput({"Base B already created."});

        return true;
    }
    else
    {
        return false;
    }
}

void Glpa::Console::CmdBase::CmdCreate::createBaseA()
{
    // Create an instance of a class that has the Glpa base class as its base class. Create windows and scenes in this class.
    ptBaseA = new ExampleBaseA();
    ptBaseA->SetName("Example A");
    ptBaseA->window->SetName(L"Example Base A");
    ptBaseA->window->SetApiClassName(L"example_base_a");
    ptBaseA->window->deleteViewStyle(WS_MAXIMIZEBOX);

    // Register the instance of the created class in glpa lib. This allows you to create windows and draw scenes.
    GlpaLib::AddBase(ptBaseA);

    // Create a window from the information set in the function of the created class instance.
    GlpaLib::CreateWindowNotApi(ptBaseA);
    // Display the created window. You can also change the display format.
    GlpaLib::ShowWindowNotApi(ptBaseA, SW_SHOWDEFAULT);

    // Load the first scene you set.
    GlpaLib::Load(ptBaseA);
}

void Glpa::Console::CmdBase::CmdCreate::createBaseB()
{
    ptBaseB= new ExampleBaseB();
    ptBaseB->SetName("Example B");
    ptBaseB->window->SetName(L"Example Base B");
    ptBaseB->window->SetApiClassName(L"example_base_b");

    ptBaseB->window->SetWidth(1920);
    ptBaseB->window->SetHeight(1080);

    GlpaLib::AddBase(ptBaseB);

    GlpaLib::CreateWindowNotApi(ptBaseB);
    GlpaLib::ShowWindowNotApi(ptBaseB, SW_SHOWDEFAULT);

    GlpaLib::Load(ptBaseB);
}
