# Basketball Tracking 🏀⛹🏻‍♀️⛹🏿‍♂️

Created by [Brett Fazio](http://linkedin.com/in/brett-fazio/) and [William Chen](https://www.linkedin.com/in/william-chen-6474a216b/)

![](assets/bron.gif)

![](assets/davis.gif)

## Overview

## Requirements 

The libraries to run the code are [cv2](https://pypi.org/project/opencv-python/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [h5py](https://www.h5py.org/) (if trying to run/evaluate on the A2D dataset). 

An extended version of cv2, ```opencv-contrib-python```, is required. Make sure this is the only cv2 package installed. ```opencv-python``` is a different package that does not include support for the trackers. Do not install multiple different opencv packages in the same environment.
```
pip install opencv-contrib-python
```

Additionally access to the YOLO tracker is required but this is already included in the `/src/yolo` folder. However, you must download the weights for the YOLO model. It can be done as follows:

```
cd src/yolo/weights/
bash download_weights.sh
```

If you wish to use the GOTURN tracker instead of the CSRT tracker (we recommend CSRT) you must download the GOTURN model [here](https://github.com/Mogball/goturn-files).

To run on the A2D dataset, the Release of the dataset itself is also required. It is available [here](https://web.eecs.umich.edu/~jjcorso/r/a2d/) and the unzipped folder entitled `Release` should be placed in the `/a2d` directory.

## Usage

The main entry point for this project is `main.py`. To avoid errors, please run it from the `src` directory. 

The most basic usage for the project would be to run on a single input video. It can be done as follows:

```
python3 main.py --video PATH
```

Where `PATH` is a path to a video file, for example:

```
python3 main.py --video ../sample_data/lebron_on_court.mp4
```

Adding the `--fast` flag only tracks the ball in frames after the first detection. 
```
python3 main.py --video ../sample_data/lebron_on_court.mp4 --fast
```

Adding the `--live` flag allows for real-time tracking. Live tracking is only available when used with the `--fast` flag.
Note: performance may be vary depending on CPU/GPU.
```
python3 main.py --video ../sample_data/lebron_on_court.mp4 --fast --live
```

### Forward Pass Only
![](assets/forwards.gif) 

### Track Backwards + Forwards
![](assets/full.gif)

## References / Credit

This project builds on the work of eriklindernoren's PyTorch Yolo implementation as a base, specifically the pre-trained model. The repo can be found [here](https://github.com/eriklindernoren/PyTorch-YOLOv3).
