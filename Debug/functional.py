'''
function for debug
'''
from __future__ import print_function

def showMessage(fileName=None, className=None, functionName=None, lineNumber=None, variableName=None, variableValue=None, message=None):
    print('DEBUG MESSAGE:', end='')
    if fileName is not None:
        print(' File:', fileName, end='')
    if className is not None:
        print(' Class: ', className, end='')
    if functionName is not None:
        print(' Function: ', functionName, end='')
    if lineNumber is not None:
        print(' Line: ', lineNumber, end='')
    if (variableName is not None) and (variableValue is not None):
        print(' (', variableName, ': ', variableValue, ')', end='')
    if message is not None:
        print(message, end='')
    print('')
    
def outputMapWrite(fileId, outputMap):
    try:
        _, h, w = outputMap.shape
        fileId.write('\t')
        for k in range(0, w):
            fileId.write('\t\t'+str(k)+'\t\t')
        fileId.write('\n')
        for j in range(0, h):
            fileId.write('\t'+str(j)+'\t\t')
            for k in range(0, w):
                fileId.write('%.3f/%.3f\t'%(outputMap[0][j][k], outputMap[1][j][k]))
            fileId.write('\n')
        fileId.write('\n')
    except:
        h, w = outputMap.shape
        for k in range(0, w):
            fileId.write('\t'+str(k))
        fileId.write('\n')
        for j in range(0, h):
            fileId.write(str(j))
            for k in range(0, w):
                fileId.write('\t%d'%(outputMap[j][k]))
            fileId.write('\n')
        fileId.write('\n')
