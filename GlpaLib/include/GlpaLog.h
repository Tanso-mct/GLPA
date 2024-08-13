#ifndef GLPA_VS_H_
#define GLPA_VS_H_

#include <Windows.h>
#include <string>

namespace Glpa
{

constexpr bool OUTPUT_DEBUG_LOG = true;
constexpr bool OUTPUT_RUNTIME_ERROR_LOG = true;
constexpr bool OUTPUT_ERROR_LOG = true;

constexpr bool OUTPUT_FILE_INFO_ENABLE = true;

constexpr bool OUTPUT_TAG_FILTER = true;

constexpr const char* OUTPUT_TAG_GLPA_LIB = "tag_glpa_lib";
constexpr const char* OUTPUT_TAG_GLPA_LIB_FRAME = "tag_glpa_lib_frame";
constexpr const char* OUTPUT_TAG_CONSOLE = "tag_console";
constexpr const char* OUTPUT_TAG_EXAMPLE = "tag_example";

constexpr const char* OUTPUT_TAG_ENABLE_FILTERS[] 
= {
    OUTPUT_TAG_GLPA_LIB,
    // OUTPUT_TAG_GLPA_LIB_FRAME
};


inline void OutputLog(const char* file, int line, const char* func, std::string tag, std::string logMsg)
{
    static int LOG_COUNT = 0;
    std::string funcStr = func;
    std::string funcName = funcStr.substr(0, funcStr.find_first_of("("));

    std::string funcType = funcName.substr(0, funcName.find_first_of(" "));
    std::string funcSig = funcName.substr(funcName.find_last_of(" ") + 1);

    std::string outputFun = funcType + " " + funcSig;

    if(OUTPUT_DEBUG_LOG)
    {
        if(OUTPUT_TAG_FILTER)
        {
            for(auto& tagFilter : OUTPUT_TAG_ENABLE_FILTERS)
            {
                if(tag == tagFilter)
                {
                    if (OUTPUT_FILE_INFO_ENABLE)
                    {
                        std::string outputStr 
                        = "LOG " + std::to_string(LOG_COUNT) + " [" + std::string(file) + ":" + std::to_string(line) + "] - " + outputFun + " - " + logMsg + "\n";
                        OutputDebugStringA(outputStr.c_str());
                    }
                    else
                    {
                        std::string outputStr 
                        = "LOG " + std::to_string(LOG_COUNT) + " - " + outputFun + " - " + logMsg + "\n";
                        OutputDebugStringA(outputStr.c_str());
                    }
                    LOG_COUNT++;
                    break;
                }
            }
        }
        else
        {
            if (OUTPUT_FILE_INFO_ENABLE)
            {
                std::string outputStr 
                = "LOG " + std::to_string(LOG_COUNT) + " [" + std::string(file) + ":" + std::to_string(line) + "] - " + outputFun + " - " + logMsg + "\n";
                OutputDebugStringA(outputStr.c_str());
            }
            else
            {
                std::string outputStr 
                = "LOG " + std::to_string(LOG_COUNT) + " - " + outputFun + " - " + logMsg + "\n";
                OutputDebugStringA(outputStr.c_str());
            }
            LOG_COUNT++;
        }
    }
}

}


#endif GLPA_VS_H_