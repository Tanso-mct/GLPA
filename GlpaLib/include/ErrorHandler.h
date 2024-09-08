#ifndef GLPA_ERROR_HANDLER_H_
#define GLPA_ERROR_HANDLER_H_

#include <string>
#include <stdexcept>
#include <Windows.h>
#include <initializer_list>

#include "GlpaLog.h"

namespace Glpa
{

inline void runTimeError(const char* file, int line, std::string errorMsg) 
{
    std::string outputStr 
    = "[" + std::string(file) + ":" + std::to_string(line) + "]" + "GlpaLib ERROR " + " - " + errorMsg + "\n";
    if (Glpa::OUTPUT_RUNTIME_ERROR_LOG) OutputDebugStringA(outputStr.c_str());
    throw std::runtime_error(outputStr.c_str());
}

inline void outputErrorLog(const char* file, int line, std::string errorMsg)
{
    if (!Glpa::OUTPUT_ERROR_LOG) return;
    std::string outputStr 
    = "[" + std::string(file) + ":" + std::to_string(line) + "]" + "GlpaLib ERROR " + " - " + errorMsg + "\n";
    OutputDebugStringA(outputStr.c_str());
}

inline void runTimeError(const char* file, int line, std::initializer_list<std::string> errorMsg) 
{
    std::string outputStr 
    = "[" + std::string(file) + ":" + std::to_string(line) + "]" + "GlpaLib ERROR " + " - ";
    for(auto& str : errorMsg)
    {
        outputStr += str + "\n";
    }
    outputStr += "\n";
    if (Glpa::OUTPUT_RUNTIME_ERROR_LOG) OutputDebugStringA(outputStr.c_str());

    throw std::runtime_error(outputStr.c_str());
}

inline void outputErrorLog(const char* file, int line,std::initializer_list<std::string> errorMsg)
{
    if (!Glpa::OUTPUT_ERROR_LOG) return;
    std::string outputStr 
    = "[" + std::string(file) + ":" + std::to_string(line) + "]" + "GlpaLib ERROR " + " - ";
    for(auto& str : errorMsg)
    {
        outputStr += str + "\n";
    }
    outputStr += "\n";

    OutputDebugStringA(outputStr.c_str());
}

}

#endif GLPA_ERROR_HANDLER_H_