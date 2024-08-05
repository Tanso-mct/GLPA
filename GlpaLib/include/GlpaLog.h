#ifndef GLPA_VS_H_
#define GLPA_VS_H_

#include <Windows.h>
#include <string>

namespace Glpa
{

constexpr bool OUTPUT_DEBUG_LOG = true;
constexpr bool OUTPUT_RUNTIME_ERROR_LOG = true;
constexpr bool OUTPUT_ERROR_LOG = true;

constexpr bool OUTPUT_TAG_FILTER = true;

constexpr const char* OUTPUT_TAG_GLPA_LIB = "tag_glpa_lib";

constexpr const char* OUTPUT_TAG_ENABLE_FILTERS[] 
= {
    OUTPUT_TAG_GLPA_LIB
};

inline void OutputLog(const char* file, int line, std::string tag, std::string logMsg)
{
    if(OUTPUT_DEBUG_LOG)
    {
        if(OUTPUT_TAG_FILTER)
        {
            for(auto& tagFilter : OUTPUT_TAG_ENABLE_FILTERS)
            {
                if(tag == tagFilter)
                {
                    std::string outputStr 
                    = "LOG [" + std::string(file) + ":" + std::to_string(line) + "]" + " - " + logMsg + "\n";
                    OutputDebugStringA(outputStr.c_str());
                    break;
                }
            }
        }
        else
        {
            std::string outputStr 
            = "LOG [" + std::string(file) + ":" + std::to_string(line) + "]" + " - " + logMsg + "\n";
            OutputDebugStringA(outputStr.c_str());
        }
    }
}

}


#endif GLPA_VS_H_