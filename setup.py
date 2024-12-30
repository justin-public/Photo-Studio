from setuptools import setup
import py2exe

setup(name="test_py2xxx",
      description="py2exe test application",
      version="0.0.1",
      windows=[{"script": "ai_main_page.py"}],
      options={
          "py2exe": {
              "includes": ["PyQt5.QtCore",
                           "PyQt5.QtWidgets",
                           "PyQt5.QtGui",
                           "PyQt5.QtCore",
                           ],
              "dll_excludes": ["msvcr71.dll",
                               "MSVCP90.dll"],
          }
      })