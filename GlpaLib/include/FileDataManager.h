#ifndef GLPA_FILE_DATA_MANAGER_H_
#define GLPA_FILE_DATA_MANAGER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>

#include "ErrorHandler.h"

#include "Polygon.h"
#include "RangeRect.cuh"
#include "Vector.cuh"

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
private :
    LPDWORD data = nullptr;

public :
    int width = 0;
    int height = 0;
    int channels = 0;

    LPDWORD getData(){return data;};
    
    void load() override;
    void release() override;
};

class ObjData : public Glpa::Data
{
private :
    std::vector<Glpa::Vec3d*> wv;
    std::vector<Glpa::Vec2d*> uv;
    std::vector<Glpa::Vec3d*> normal;

    Glpa::RangeRect* rangeRect = nullptr;
    std::vector<Glpa::Polygon*> polygons;

public :
    void load() override;
    void release() override;

    int getPolyAmount();
    void getPolyData(std::vector<Glpa::GPU_POLYGON>& polys);
    Glpa::GPU_RANGE_RECT getRangeRectData();
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
    FileDataManager();
    ~FileDataManager();
    void newFile(std::string path);
    void deleteFile(std::string path);

    void load(std::string path);
    void release(std::string path);

    std::string getFileName(std::string path);
    int getWidth(std::string path);
    int getHeight(std::string path);
    int getChannels(std::string path);

    LPDWORD getPngData(std::string path);

    int getPolyAmount(std::string path);
    void getPolyData(std::string path, std::vector<Glpa::GPU_POLYGON>& polys);
    Glpa::GPU_RANGE_RECT getRangeRectData(std::string path);


};

}

#endif GLPA_FILE_DATA_MANAGER_H_