#ifndef GLPA_FILE_DATA_MANAGER_H_
#define GLPA_FILE_DATA_MANAGER_H_

#include <string>
#include <unordered_map>

namespace Glpa 
{

class Data
{
public :
    virtual void load(){};
    virtual void release(){};
    virtual void destroy(){};
};

class PngData : public Glpa::Data
{
public :
    int width = 0;
    int height = 0;
    int channels = 0;
    LPDWORD data;

    void load() override;
    void release() override;
    void destroy() override;
};

class ObjData : public Glpa::Data
{
public :
    void load() override;
    void release() override;
    void destroy() override;
};

class FileData
{
public :
    std::string fileName;
    std::string filePath;

    Glpa::Data data;
};


class FileDataManager
{
private :
    std::unordered_map<std::string, Glpa::FileData> files;

public :
    void newFile();
    void deleteFile();

    void load();
    void release();
    void destroy();

    void getFileName();
    void getFilePath();

    bool getWidth();
    bool getHeight();
    bool getChannels();
    bool getData();
};

}

#endif GLPA_FILE_DATA_MANAGER_H_