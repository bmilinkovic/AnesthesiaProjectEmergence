# Anesthesia Emergence Project

This repository has dependencies with other git repositories that are stored as submodules in the src/ directory.
To correctly close the submodules to be used locally onn your machine run:

git clone --recurse-submodules https://github.com/bmilinkovic/AnesthesiaProjectEmergence.git

This should clone the submodules into your local git repo.



startup.m will run and load all relevant paths and directories. For startup.m to run open MATLAB from the parent directory,
i.e. the location of the startup.m script. [Note startup.m script is yet to be configured properly as it is a template used on previous projects. Feel free to alter it to suit the environment that you're working from].

