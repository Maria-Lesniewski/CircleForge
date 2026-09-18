# @author Maria Lesniewski
# @file io.py
# @desc This is a file for holding re-usable io functions employed in driver scripts
#       and creating a stable anchor for path references
import os
import numpy as np
from pathlib import Path
import sys

## USER CL INTERFACE
def yes_no_q(extra_info : str, question : str) -> bool:
    """ Asks a yes no question, returns true if yes, false if not 
        extra_info : Context paragraph ahead of question
        question : y/n question
        (we print Y or N as options and force one)
    """
    while(True):
        print(extra_info)
        print(question)
        response = input("Y/N\n")
        if response == "Y":
            return True
        if response == "N":
            return False
        else:
            print("Please input 'Y' or 'N'")



def multiple_choice_q(extra_info: str, multiple_choice : list[str]):
    """
    Asks a multiple choice question and returns the answer.
    extra_info : Context paragraph ahead of question 
    multiple_choice : Options user can select from, entry will be returned
    """
    integer_options = np.arange(0, len(multiple_choice))
    while(True):
        print(extra_info)
        print("Please select an integer option")
        print("Options:")
        for i in range(0, len(multiple_choice)): 
            print(f"{i}. {multiple_choice[i]}")
        
        response = input("Your Choice?\n")
        try:
            response = int(response)
            if (response in integer_options):
                return multiple_choice[response]
            else:
                print(f"\nSelection out of range. Please select integer in range {integer_options[0]} - {integer_options[-1]}\n")
        except:
            print(f"\n'{response}' is not an integer. Try again.\n")


## FUNCTION FOR PATH ANCHORING ####
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
