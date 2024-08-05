#ifndef GLPA_ERROR_HANDLER_H_
#define GLPA_ERROR_HANDLER_H_

#include <string>
#include <stdexcept>
#include <Windows.h>

namespace Glpa
{

inline void runTimeError(const char* file, int line, std::string errorMsg) 
{
    std::string outputStr 
    = "[" + std::string(file) + ":" + std::to_string(line) + "]" + "GlpaLib ERROR " + " - " + errorMsg + "\n";
    OutputDebugStringA(outputStr.c_str());
    throw std::runtime_error(errorMsg.c_str());
}

inline void outputErrorLog(const char* file, int line,std::string errorMsg)
{
    std::string outputStr 
    = "[" + std::string(file) + ":" + std::to_string(line) + "]" + "GlpaLib ERROR " + " - " + errorMsg + "\n";
    OutputDebugStringA(outputStr.c_str());
}

}

#endif GLPA_ERROR_HANDLER_H_