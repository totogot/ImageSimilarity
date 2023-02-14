# TEMAPLTE PROJECT

BRIEF BLURB

The main.ipynb contains the run code, with accompanying commentary, while the sma package contains helper functions and wrappers.

## Initial setup
The first thing you are going to want to do is set up a virtual environment for installing all package requirements into

```
$ cd C:\Users\jdoe\Documents\PersonalProjects\TEMPLATE
$ python -m venv venv
```

Then from within the terminal command line within your IDE (making sure you are in the project folder), you can install all the dependencies for the project, by simply activating the venv and leveraging the setuptools package and the setup.cfg file created in the project repo. 
Note: for IDEs where the CLI uses PowerShell by default (e.g. VS Code), in order to run Activate.ps1 you may find that you first need to update your settings such that Command Prompt is the default terminal shell - see here: https://support.enthought.com/hc/en-us/articles/360058403072-Windows-error-activate-ps1-cannot-be-loaded-because-running-scripts-is-disabled-UnauthorizedAccess-

```
$ .\venv\Scripts\activate
$ pip install --upgrade pip
$ pip install .
```

This last command will install all dependencies outlined in the setup.cfg file. ipykernel has been included to enable the main.ipynb to be run also and for relevant visualisations to be outputted also.


## Data


