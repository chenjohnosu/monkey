# check_nltk_conflict.py
import os
import sys

# Check current directory for nltk.py
current_dir = os.getcwd()
nltk_py_in_current = os.path.exists(os.path.join(current_dir, 'nltk.py'))

if nltk_py_in_current:
    print(f"WARNING: Found 'nltk.py' in current directory: {current_dir}")
    print("This file is conflicting with the real NLTK library.")
    print("Please rename this file to something else like 'nltk_helper.py'")
else:
    print("No 'nltk.py' file found in current directory.")

# Try to import nltk and see where it's coming from
try:
    import nltk
    print(f"NLTK imported from: {nltk.__file__}")
except ImportError:
    print("Could not import NLTK. Is it installed?")
except AttributeError as e:
    print(f"AttributeError when importing NLTK: {e}")
    print("This suggests your NLTK installation is broken or being shadowed.")