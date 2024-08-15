#ifndef GLPA_VS_H_
#define GLPA_VS_H_

#include <Windows.h>
#include <string>
#include <chrono>

namespace Glpa
{

constexpr bool OUTPUT_DEBUG_LOG = true;
constexpr bool OUTPUT_RUNTIME_ERROR_LOG = true;
constexpr bool OUTPUT_ERROR_LOG = true;

constexpr bool OUTPUT_FILE_INFO_ENABLE = true;

constexpr bool OUTPUT_TAG_FILTER = true;

constexpr const char* OUTPUT_TAG_GLPA_LIB = "tag_glpa_lib";
constexpr const char* OUTPUT_TAG_GLPA_LIB_FRAME = "tag_glpa_lib_frame";

constexpr const char* OUTPUT_TAG_GLPA_FILE_DATA_MG = "tag_glpa_file_data_manager";
constexpr const char* OUTPUT_TAG_GLPA_BASE = "tag_glpa_base";
constexpr const char* OUTPUT_TAG_GLPA_SCENE = "tag_glpa_scene";
constexpr const char* OUTPUT_TAG_GLPA_SCENE_2D = "tag_glpa_scene_2d";
constexpr const char* OUTPUT_TAG_GLPA_SCENE_3D = "tag_glpa_scene_3d";
constexpr const char* OUTPUT_TAG_GLPA_RENDER = "tag_glpa_render";

constexpr const char* OUTPUT_TAG_CONSOLE = "tag_console";
constexpr const char* OUTPUT_TAG_EXAMPLE = "tag_example";

constexpr const char* OUTPUT_TAG_ENABLE_FILTERS[] 
= {
    OUTPUT_TAG_GLPA_LIB,
    OUTPUT_TAG_GLPA_RENDER,
    // OUTPUT_TAG_GLPA_LIB_FRAME
};


inline void OutputLog(const char* file, int line, const char* func, std::string tag, std::string logMsg)
{
    static int LOG_COUNT = 0;
    static auto lastTime = std::chrono::system_clock::now();

    auto now = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed = now - lastTime;
    float elapsedTime = elapsed.count();
    bool isNewLine = elapsedTime > 0.5;

    std::string funcStr = func;
    std::string funcName = funcStr.substr(0, funcStr.find_first_of("("));

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
                        = "LOG " + std::to_string(LOG_COUNT) + " [" + std::string(file) + ":" + std::to_string(line) + "] - " + funcName + " - " + logMsg + "\n";
                        if (isNewLine) OutputDebugStringA("\n");
                        OutputDebugStringA(outputStr.c_str());
                        lastTime = now;
                    }
                    else
                    {
                        std::string outputStr 
                        = "LOG " + std::to_string(LOG_COUNT) + " - " + funcName + " - " + logMsg + "\n";
                        if (isNewLine) OutputDebugStringA("\n");
                        OutputDebugStringA(outputStr.c_str());
                        lastTime = now;
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
                = "LOG " + std::to_string(LOG_COUNT) + " [" + std::string(file) + ":" + std::to_string(line) + "] - " + funcName + " - " + logMsg + "\n";
                if (isNewLine) OutputDebugStringA("\n");
                OutputDebugStringA(outputStr.c_str());
                lastTime = now;
            }
            else
            {
                std::string outputStr 
                = "LOG " + std::to_string(LOG_COUNT) + " - " + funcName + " - " + logMsg + "\n";
                if (isNewLine) OutputDebugStringA("\n");
                OutputDebugStringA(outputStr.c_str());
                lastTime = now;
            }
            LOG_COUNT++;
        }
    }
}

}


#endif GLPA_VS_H_