# panoramaProject
csci576 panorama final Project

## environment setup
```python3 -m venv ~/cs576venv```\
```source ~/cs576venv/bin/activate```\
```pip3 install -r requirements.txt```

## Usage
```python3 main.py -f <video file path> [-fg] [foreground extraction method]```
- Add `-c` to remove cached file if a new panorama is needed

### example
```python3 main.py -f sample.mp4 -fg mog2```