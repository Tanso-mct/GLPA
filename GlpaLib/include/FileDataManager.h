#ifndef GLPA_FILE_DATA_MANAGER_H_
#define GLPA_FILE_DATA_MANAGER_H_

#include <string>
#include <unordered_map>

#include "ErrorHandler.h"

namespace Glpa 
{

class Data
{
public :
    std::string fileName;
    std::string filePath;
    std::string extension;

    virtual void load(){};
    virtual void release(){};
};

class PngData : public Glpa::Data
{
public :
    int width = 0;
    int height = 0;
    int channels = 0;
    LPDWORD data = nullptr;

    void load() override;
    void release() override;
};

class ObjData : public Glpa::Data
{
public :
    void load() override;
    void release() override;
};

class FileData
{
public :
    Glpa::Data* data;
};


class FileDataManager
{
private :
    std::unordered_map<std::string, Glpa::FileData*> files;

public :
    ~FileDataManager();
    void newFile(std::string path);
    void deleteFile(std::string path);

    void load(std::string path);
    void release(std::string path);

    std::string getFileName(std::string path);
    int getWidth(std::string path);
    int getHeight(std::string path);
    int getChannels(std::string path);
    LPDWORD getData(std::string path);
};

}

#endif GLPA_FILE_DATA_MANAGER_H_