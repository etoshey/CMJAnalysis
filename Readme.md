# CMJ Analysis

Using this Python script, you can compute jump height, flight time, RSImod, and more. You just need to capture your force plate data in a C3D file.

# Install Python Packages

```python
    pip install pyc3dtools matplotlib rich
```

# Run script

First, copy your token from <a  href="C3dtools.com"> C3DTools </a> and past it in the main.py, then run script file

```python
    python main.py --arg1 C3DFilePath --arg2 ForceplateNumber

    python main.py --arg1 ./data/FPCMJ.c3d --arg2 2 #Sample file

```

If the C3D file is read correctly, a plot will open. You can select the relevant zone by dragging. Just remember that the subject's mass is calculated based on the first 100 sample data points.</br>

![CMJ](doc/1.png)

Processed signal:</br>

![CMJ](doc/2.png)

Now you can see the results in terminal:
</br>
</br>

![CMJ](doc/3.png)
