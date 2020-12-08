# Basketball Tracking 🏀⛹🏻‍♀️⛹🏿‍♂️

Created by [Brett Fazio](http://linkedin.com/in/brett-fazio/) and [William Chen](https://www.linkedin.com/in/william-chen-6474a216b/)

![](assets/bron.gif)

![](assets/davis.gif)

## Overview

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

### Forward Pass Only
![](assets/forwards.gif) 

### Track Backwards + Forwards
![](assets/full.gif)

## References / Credit

This project builds on the work of eriklindernoren's PyTorch Yolo implementation as a base, specifically the pre-trained model. The repo can be found [here](https://github.com/eriklindernoren/PyTorch-YOLOv3).
