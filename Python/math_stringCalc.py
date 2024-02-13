
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
            return leftSt + " * " + rightSt


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


mtRotZ = {
    1 : "cosRotZ",  2  : "-sinRotZ", 3  : "0", 4  : "0",
    5 : "sinRotZ",  6  : "cosRotZ",  7  : "0", 8  : "0",
    9 : "0",        10 : "0",        11 : "1", 12 : "0",
    13 : "0",       14 : "0",        15 : "0", 16 : "1"
}

mtRotY = {
    1 : "cosRotY",   2  : "0", 3  : "sinRotY", 4  : "0",
    5 : "0",         6  : "1", 7  : "0",       8  : "0",
    9 : "-sinRotY",  10 : "0", 11 : "cosRotY", 12 : "0",
    13 : "0",        14 : "0", 15 : "0",       16 : "1"
}

mtRotX = {
    1 : "1",  2  : "0",       3  : "0",        4  : "0",
    5 : "0",  6  : "cosRotX", 7  : "-sinRotX", 8  : "0",
    9 : "0",  10 : "sinRotX", 11 : "cosRotX",  12 : "0",
    13 : "0", 14 : "0",       15 : "0",        16 : "1"
}

mtTrans = {
    1 : "1",  2  : "0", 3  : "0", 4  : "transX",
    5 : "0",  6  : "1", 7  : "0", 8  : "transY",
    9 : "0",  10 : "0", 11 : "1", 12 : "transZ",
    13 : "0", 14 : "0", 15 : "0", 16 : "1"
}

mtRotXTrans = mt4x4Product(mtRotX, mtTrans)
mtRotYRotXTrans = mt4x4Product(mtRotY, mtRotXTrans)
mtRotZRotYRotXTrans = mt4x4Product(mtRotZ, mtRotYRotXTrans)

result = mtRotZRotYRotXTrans


print(result)

# {1: 'cosRotY',  2: 'sinRotY * sinRotX',  3: 'sinRotY * cosRotX',  4: '0', 
#  5: '0',        6: 'cosRotX',            7: '-sinRotX',           8: '0', 
#  9: '-sinRotY',10: 'cosRotY * sinRotX', 11: 'cosRotY * cosRotX', 12: '0', 
#  13: '0',      14: '0',                 15: '0',                 16: '1'}

{1: 'cosRotZ * cosRotY', 2: 'cosRotZ * sinRotY * sinRotX * -sinRotZ * cosRotX', 3: 'cosRotZ * sinRotY * cosRotX * -sinRotZ * -sinRotX', 4: 'cosRotZ * cosRotY * transX * sinRotY * sinRotX * transY * cosRotX * transZ * -sinRotZ * cosRotX * transY * -sinRotX * transZ', 
 5: 'sinRotZ * cosRotY', 6: 'sinRotZ * sinRotY * sinRotX * cosRotZ * cosRotX',  7: 'sinRotZ * sinRotY * cosRotX * cosRotZ * -sinRotX',  8: 'sinRotZ * cosRotY * transX * sinRotY * sinRotX * transY * cosRotX * transZ * cosRotZ * cosRotX * transY * -sinRotX * transZ', 
 9: '-sinRotY',         10: 'cosRotY * sinRotX',                               11: 'cosRotY * cosRotX',                                12: '-sinRotY * transX * cosRotY * sinRotX * transY * cosRotX * transZ', 
 13: '0',               14: '0',                                               15: '0',                                                16: '1'}