'''
Script to print with colors in terminal without having to remember the ANSI codes
'''

def printRed(message):
    print("\033[91m{}\033[0m".format(message))

def printGreen(message):
    print("\033[92m{}\033[0m".format(message))

def printOrange(message):
    print("\033[93m{}\033[0m".format(message))

def printBlue(message):
    print("\033[94m{}\033[0m".format(message))

