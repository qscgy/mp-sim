# MP-Sim
This is a simulator for running motion profiles on robots. Right now, it only supports tank drive motion profiles
in the same format as those created by Jaci's Pathfinder. It uses the characterization parameters desribed in [this whitepaper](https://www.chiefdelphi.com/media/papers/3402) to simulate PIDF control of a robot following a motion profile.

## Dependencies
Required Python libraries are Numpy, Scipy, Pandas, and Matplotlib. This software was developed under Python 3.5,
Numpy 1.14.2, Scipy 1.0.1, Pandas 0.22.0, and Bokeh 0.12.

## How to Use
First, install Bokeh. The easiest way to do this is via pip, with `pip install bokeh`. Once you're done this, you will need to make sure that the install location's `bin` directory is in your $PATH. On my computer (running Debian 9), packages install at `~/.local`, so I would run `export PATH=$PATH:~/.local/bin` from the command line.
Once you've set up bokeh, download all files for MP-Sim and run `bokeh serve simulator.py` from the directory containing the files. Right now, in order to change the files, you need to edit the file names in `simulator.py`.
