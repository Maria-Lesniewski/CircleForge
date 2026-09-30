# @author Maria Lesniewski
# @file type_check.py
# @desc This is a helper file for type checking user io

def is_int(response):
    "checks integer conversion"
    try:
        int(response)
        return True
    except ValueError:
        return False

def is_float(response):
    "checks float conversion"
    try:
        float(response)
        return True
    except ValueError:
        return False
