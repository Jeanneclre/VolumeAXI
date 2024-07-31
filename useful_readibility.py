'''
Script to print with colors in terminal without having to remember the ANSI codes
'''

def printRed(message:str,obj=None):
    '''
    function working as print (with a maximum of 2 parameters) but with red color
    '''
    if obj is not None:
        print("\033[91m{}".format(message),obj,"\033[0m")
    else:
        print("\033[91m{}\033[0m".format(message))

def printGreen(message:str,obj=None):
    if obj is not None:
        print("\033[92m{}".format(message),obj,"\033[0m")
    else:
        print("\033[92m{}\033[0m".format(message))

def printOrange(message:str,obj=None):
    if obj is not None:
        print("\033[93m{}".format(message),obj,"\033[0m")
    else:
        print("\033[93m{}\033[0m".format(message))

def printBlue(message:str,obj=None):
    if obj is not None:
        print("\033[94m{}".format(message),obj,"\033[0m")
    else:
        print("\033[94m{}\033[0m".format(message))

