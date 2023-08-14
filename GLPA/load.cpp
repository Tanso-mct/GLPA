
#include "load.h"

int FILELOAD::loadBinary(char filePath[MAX_FILE_PATH_CHAR])
{
    std::ifstream ifs(filePath, std::ios_base::in | std::ios_base::binary);
    if (ifs.fail()) 
    {
        //ERROR
        return LOAD_FAILURE;
    }
    
    
    

}