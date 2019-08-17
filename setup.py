import sys
import os
from cx_Freeze import setup, Executable

os.environ['TCL_LIBRARY'] = r"C:\Users\perks\Miniconda3\envs\py35\tcl\tcl8.6"
os.environ['TK_LIBRARY'] = r"C:\Users\perks\Miniconda3\envs\py35\tcl\tk8.6"

build_exe_options = {"packages": ["os", "numpy"], "includes": ["numpy"]}
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(name="Stanny",
      version="0.1",
      description="My GUI application!",
      options={"build_exe": build_exe_options},
      executables=[Executable("Stanny.py", base=base)])