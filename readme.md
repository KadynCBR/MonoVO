# MonoVO

## Usage:
Assumes you have the `dataset` folder of KITTI Odometry set in the parent folder.

Run the program using the provided driver script.
```bash
# ./runner.sh <SEQUENCE ID>
./runner.sh 00
```

Otherwise you can run the driver program directly
```
python runner.py <KITTI Image Directory> <KITTI Calibration File> <KITTI Pose File> <Num Frames> <-d>
```
removing the `-d` at the end of the command will not show any debug or live graphs.

