# @author Maria Lesniewski
# @file io.py
# @desc This is a file for holding re-usable io functions employed in driver scripts
#       and creating a stable anchor for path references
import os
import numpy as np
from pathlib import Path
import sys
import inspect
from typing import Any
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

    returns multiple_choice[response] : str 
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


def compounding_multiple_choice_q(extra_info : str, multiple_choice : list[str]):
    """
    Asks a multiple choice question where mutiple options may be admitted and returns answers

    extra_info : Context paragraph ahead of question
    multiple_choice : Options user can select from, entry will be returned

    returns multiple_choice[responses] : list[str]
    """
    integer_options = np.arange(0, len(multiple_choice))
    sentinal = -1
    while(True):
        print(extra_info)
        print(f"Please select integer options seperated by spaces, followed by {sentinal}")
        for i, item in enumerate(multiple_choice):
            print(f"{i}. {item}")
        response = input("Select options:")

        if not response:
            print("Please enter at least one option")
            continue

        try:
            responses = response.split()
            if len(responses) == 0:
                print("Please enter at least one option")
                continue
            int_responses = [int(x) for x in responses]
        except:
            print(f"\n {response} does not contain all integers. Try again \n")
            continue

        try:
            if ( int_responses[-1] == sentinal and 
                    all(x in integer_options for x in int_responses[:-1])
                   ):
                print(f"registered \n {int_responses}")
                selected = [multiple_choice[i] for i in int_responses[:-1]]
                return list(set(selected))
            else:
                print("\n Integer responses entered are out of range or formatted incorrectly")
                print(f"\n Please enter options as integers in range {integer_options[0]} - {integer_options[-1]} followed by a '-1'")
                continue

        except:
            print(f"\nCritical Logic Block Error. Killing program.\n")
            sys.exit(1)


def dictionary_filling_question(extra_info : str, keys : list[str] ) -> dict[str:Any]:
    """
    Asks a fill in the blank style question wherein user can write anything back
    (The burden of type correctness is on the calling program to check)

    extra_info : Context paragraph ahead of question
    keys : categories that user fills values for

    returns dict[keys, user_responses]
    """
    print(extra_info)
    print(f"Please fill in values for the following entries")
    fin_dict = {}
    for entry in keys:
        fin_dict[entry] = input(f"{entry}:")
    return fin_dict

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

def find_here() -> Path:
    """
    Finds the absolute path of a calling file's folder
    (NOT the calling file. the directory its in)
    """
    caller_file_directory = Path(inspect.stack()[1].filename).parent.resolve()
    return caller_file_directory
