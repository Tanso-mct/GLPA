#ifndef GLPA_FILE_H_
#define GLPA_FILE_H_

#include <string>

#include "SceneObject.h"

namespace Glpa
{

class File : public Glpa::SceneObject
{
protected :
    std::string fileName;
    std::string filePath;

public :
    File();
    ~File() override;

    std::string getFilePath() const {return filePath;}
    void setFilePath(std::string str);
    
};

}

#endif GLPA_FILE_H_
