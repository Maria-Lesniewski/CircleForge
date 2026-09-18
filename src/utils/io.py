# @author Maria Lesniewski
# @file io.py
# @desc This is a file for holding re-usable io functions employed in driver scripts
#       and creating a stable anchor for path references
import os
from pathlib import Path
import sys

def yes_no_q(extra_info : str, question : str) -> bool:
    """ Asks a yes no question, returns true if yes, false if not 
        extra_info : Context paragraph ahead of question
        question : y/n question
        (we print Y or N as options and force one)
    """
    while(True):
        print(extra_info)
        print(question)
        response = input("Y/N")
        if response == "Y":
            return True
        if response == "N":
            return False
        else:
            print("Please input 'Y' or 'N'")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"

def make_artifact_dir(name : str) -> Path:
    """
    finds artifacts in the root folder, makes folder <name>
    """
    path = ARTIFACTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def artifact_path(name : str) -> Path:
    """
    constructs a filename path including artifact root
    """
    path = ARTIFACTS / name 
    return path

def find_project_root() -> Path:
    """
    returns PROJECT_ROOT
    """
    return PROJECT_ROOT

def delete_file_abs_path(abs_file_path : str) -> bool: 
    """
    removes a file using the passed absolute path as an arg
    returns true if successful, false if not
    """
    if (Path(abs_file_path).exists()):
        Path(abs_file_path).unlink()
    return (not Path(abs_file_path).exists())
