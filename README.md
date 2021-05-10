# SELF DRIVING CAR

## REQUIREMENTS

### PYTHON 2.7.13
### PYTHON LIBS ON TERMINAL

Dependencies are in `requirements.txt`

1. To install the dependencies, first go to self-driving-car directory, once in, run:

```
$ pip install -r requirements.txt
```

2. Afterwards, you can run the python file `map.py` with:

```
$ python map.py
```

## EXPLANATION

The file `map.py` contains the environment where the user is able to draw sand onto the map, and it generates the car from `car.kv`. It also imports the deep q-learning neural network from `ai.py` and uses it to try to find a way to go from the top-left of the screen to the bottom-right of the screen (continuously alternating between these 2 positions). You are also able to clear the screen in case you make a mistake while drawing the sand. Furthermore, you can save your brain once you are done allowing it to explore, and the brain is saved in `last_brain.pth` (note that `map.py` quits once the brain has been saved). This file is also called to import the brain when the load button is clicked - loads the brain.
