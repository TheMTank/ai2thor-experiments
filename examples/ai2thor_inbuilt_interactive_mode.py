"""
Control ai2thor in interactive mode with a unity build path. Needs to be run from terminal.
If you don't provide a real path, it will try to download the 600+mb ai2thor assets.

Only works from terminal!
"""

import argparse
import os

import ai2thor.controller

parser = argparse.ArgumentParser(description='Run ai2thor in interactive mode')
parser.add_argument('--unity-build-name', type=str,
                    default='build_bowls_vs_cups_fp1_201_301_4011_v_0.1.x86_64',
                    help='Path to a unity build with file ending in .x86_64')
args = parser.parse_args()

controller = ai2thor.controller.Controller()
# file must be in gym_ai2thor/build_files
unity_build_abs_file_path = os.path.abspath(os.path.join(__file__, '../../gym_ai2thor/build_files',
                                                         args.unity_build_name))
print('Build file path at: {}'.format(unity_build_abs_file_path))
if not os.path.exists(unity_build_abs_file_path):
    raise ValueError('Unity build file does not exist')
controller.local_executable_path = unity_build_abs_file_path
controller.start()

controller.reset('FloorPlan201')
controller.step(dict(action='Initialize', gridSize=0.05, cameraY=-0.8, continuous=True))
controller.interact()
