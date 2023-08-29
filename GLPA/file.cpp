// #define DEBUG_LOAD_
#include "file.h"

int FILELOAD::loadBinary(int fileType, std::string inputFileName)
{
    std::string filePath ("../x64/Debug/");
    fileName = inputFileName;

    switch (fileType)
    {
    case FILETYPE_BMP :
        filePath.append("bmp");
        break;
    
    case FILETYPE_PNG :
        filePath.append("png");
        break;
    
    default:
        break;
    }

    filePath.append("/");
    filePath.append(fileName);
    
    //ファイル名からバイナリファイルで読み込む
    std::ifstream file(filePath, std::ios::binary);
    if (file.fail())
    {
        #ifdef DEBUG_LOAD_

        OutputDebugStringA("file open failed\n");

        #endif
        return 1;
    }

    //読込サイズを調べる。
    file.seekg(0, std::ios::end);
    long long int size = file.tellg();
    file.seekg(0);
    std::vector<std::string> tempBinaryData (size);

    //読み込んだデータをchar型に出力する
    char *fileData = new char[size];
    file.read(fileData, size);

    #ifdef DEBUG_LOAD_

    //サイズを出力する
    OutputDebugStringA(("size = " + std::to_string(size) + "\n").c_str());

    #endif p

    char hexChar[9];
    int decimal;

    //バイナリデータの格納
    for (int i = 1; i < size + 1; i++)
    {
        decimal = std::stoi(std::to_string(fileData[i - 1]));
        // char hex_str[9];
        sprintf_s(hexChar, sizeof(hexChar),"%.2X", decimal);
        std::string hexString(hexChar);
        if (std::stoi(std::to_string(fileData[i - 1])) < 0)
        {
            hexString.erase(0, 6);
            tempBinaryData[i - 1] = hexString;
        }
        else
        {
            tempBinaryData[i - 1] = hexString;
        }
    }
    
    binaryData.swap(tempBinaryData);

    delete fileData;

    #ifdef DEBUG_LOAD_

    OutputDebugStringA("END");

    #endif
    return 0;
}

void FILELOAD::checkBinary()
{
    //16進数バイナリデータを表示する
    for (int i = 1; i < binaryData.size() + 1; i++)
    {
        OutputDebugStringA("   ");
        OutputDebugStringA((binaryData[i - 1]).c_str());

        if ((i % 16) == 0)
        {
            OutputDebugStringA("\n");
        }
    }
}

int BMP_FILE::readBinary()
{
    return 0;
}

int OBJ_FILE::readBinary(int fileType, std::string inputFileName)
{
    std::string filePath ("../x64/Debug/");
    fileName = inputFileName;

    switch (fileType)
    {
    case FILETYPE_BMP :
        filePath.append("bmp");
        break;
    
    case FILETYPE_PNG :
        filePath.append("png");
        break;
    
    default:
        break;
    }

    filePath.append("/");
    filePath.append(fileName);

    std::ifstream file(filePath);

    if (file.fail())
    {
        #ifdef DEBUG_LOAD_

        OutputDebugStringA("file open failed\n");

        #endif
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        
    }

    file.close();

    return 0;
}

BMP_FILE sampleBmpFile;
