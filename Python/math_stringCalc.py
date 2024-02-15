
def add(leftSt, rightSt) :
    if leftSt == "0"  or leftSt == "" :
        if rightSt == "0"  or rightSt == ""  :
            return ""
        
        else :
            return rightSt
        
    else :
        if rightSt == "0"  or rightSt == ""  :
            return leftSt
        
        else :
            return leftSt + " + " + rightSt


def add4Value(stringA, stringB, stringC, stringD) :
    aPlusB = add(stringA, stringB)
    cPlusD = add(stringC, stringD)

    return add(aPlusB, cPlusD)


def product(stringA, stringB) :
    if stringA == "0"  or stringA == "" :
        return ""
    
    if stringB == "0"  or stringB == ""  :
        return ""
    
    if stringA == "1" :
        return stringB
    
    if stringB == "1" :
        return stringA

    return stringA + " * " + stringB


MATRIX_SAMPLE = {
    1 : "",  2  : "", 3  : "", 4  : "",
    5 : "",  6  : "", 7  : "", 8  : "",
    9 : "",  10 : "", 11 : "", 12 : "",
    13 : "", 14 : "", 15 : "", 16 : ""
}


def mt4x4OneResult(leftKeyStNum, rightKeyStNum, leftMt, rightMt) :
    rtNum = add4Value(
        product(leftMt[leftKeyStNum], rightMt[rightKeyStNum]), product(leftMt[leftKeyStNum+1], rightMt[rightKeyStNum+4]), 
        product(leftMt[leftKeyStNum+2], rightMt[rightKeyStNum+8]), product(leftMt[leftKeyStNum+3], rightMt[rightKeyStNum+12])
    )

    if rtNum == "" :
        return "0"
    
    return rtNum

def mt4x4Product(leftMt, rightMt) :
    rtMt = {
        1:mt4x4OneResult(1, 1, leftMt, rightMt),  2:mt4x4OneResult(1, 2, leftMt, rightMt), 3:mt4x4OneResult(1, 3, leftMt, rightMt), 4:mt4x4OneResult(1, 4, leftMt, rightMt),
        5:mt4x4OneResult(5, 1, leftMt, rightMt),  6 :mt4x4OneResult(5, 2, leftMt, rightMt), 7 :mt4x4OneResult(5, 3, leftMt, rightMt), 8 :mt4x4OneResult(5, 4, leftMt, rightMt),
        9:mt4x4OneResult(9, 1, leftMt, rightMt),  10:mt4x4OneResult(9, 2, leftMt, rightMt), 11:mt4x4OneResult(9, 3, leftMt, rightMt), 12:mt4x4OneResult(9, 4, leftMt, rightMt),
        13:mt4x4OneResult(13, 1, leftMt, rightMt), 14:mt4x4OneResult(13, 2, leftMt, rightMt), 15:mt4x4OneResult(13, 3, leftMt, rightMt), 16:mt4x4OneResult(13, 4, leftMt, rightMt)
    }

    return rtMt


def printMt(mt) :
    for key, stringNum in mt.items():
        print("---" , key - 1 , "---")
        print(stringNum)


mtRotX = {
    1 : "1",  2  : "0",       3  : "0",        4  : "0",
    5 : "0",  6  : "cos(RAD(rot.x))", 7  : "-sin(RAD(rot.x))", 8  : "0",
    9 : "0",  10 : "sin(RAD(rot.x))", 11 : "cos(RAD(rot.x))",  12 : "0",
    13 : "0", 14 : "0",       15 : "0",        16 : "1"
}

mtRotY = {
    1 : "cos(RAD(rot.y))",   2  : "0", 3  : "sin(RAD(rot.y))", 4  : "0",
    5 : "0",         6  : "1", 7  : "0",       8  : "0",
    9 : "-sin(RAD(rot.y))",  10 : "0", 11 : "cos(RAD(rot.y))", 12 : "0",
    13 : "0",        14 : "0", 15 : "0",       16 : "1"
}

mtRotZ = {
    1 : "cos(RAD(rot.z))",  2  : "-sin(RAD(rot.z))", 3  : "0", 4  : "0",
    5 : "sin(RAD(rot.z))",  6  : "cos(RAD(rot.z))",  7  : "0", 8  : "0",
    9 : "0",        10 : "0",        11 : "1", 12 : "0",
    13 : "0",       14 : "0",        15 : "0", 16 : "1"
}

mtTrans = {
    1 : "1",  2  : "0", 3  : "0", 4  : "trans.x",
    5 : "0",  6  : "1", 7  : "0", 8  : "trans.y",
    9 : "0",  10 : "0", 11 : "1", 12 : "trans.z",
    13 : "0", 14 : "0", 15 : "0", 16 : "1"
}

a = mt4x4Product(mtRotY, mtRotX)
b = mt4x4Product(mtRotZ, a)

result = b


printMt(result)