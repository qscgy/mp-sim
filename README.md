# MP-Sim
This is a simulator for running motion profiles on robots. Right now, it only supports tank drive motion profiles
in the same format as those created by Jaci's Pathfinder. It uses the characterization parameters desribed in [this whitepaper](https://www.chiefdelphi.com/media/papers/3402) to simulate PIDF control of a robot following a motion profile.

## Dependencies
Required Python libraries are Numpy, Pandas, and Matplotlib. This software was developed with Python 3.7.2, but should work with any Python 3.x.

## How to Use
Run `simulator.py` from the command line or your favorite IDE or text editor. As of now, you need to edit the dictionary at the bottom of `simulator.py` to change constants and filenames.
