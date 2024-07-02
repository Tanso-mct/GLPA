#ifndef GLPA_ERROR_HANDLER_H_
#define GLPA_ERROR_HANDLER_H_

#include <string>
#include <stdexcept>
#include <Windows.h>

namespace Glpa
{

inline void runTimeError(std::string place, std::string errorMsg) 
{
    std::string outputStr = "GlpaLib ERROR " + place + " - " + errorMsg + "\n";
    OutputDebugStringA(outputStr.c_str());
    throw std::runtime_error(errorMsg.c_str());
}

inline void outputErrorLog(std::string place, std::string errorMsg)
{
    std::string outputStr = "GlpaLib ERROR " + place + " - " + errorMsg + "\n";
    OutputDebugStringA(outputStr.c_str());
}

}

#endif GLPA_ERROR_HANDLER_H_