"""
Copied and modified from
http://ai2thor.allenai.org/tutorials/examples
Added helper functions to draw bounding boxes retrieved from ai2thor
Also can test some reward functions here
"""

import numpy as np
import skimage.color, skimage.transform
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
from matplotlib import pyplot as plt
import ai2thor.controller

from gym_ai2thor.task_utils import check_if_focus_and_close_enough_to_object_type, \
    show_bounding_boxes, show_instance_segmentation


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
    # Kitchens: FloorPlan1 - FloorPlan30
    # Living rooms: FloorPlan201 - FloorPlan230
    # Bedrooms: FloorPlan301 - FloorPlan330
    # Bathrooms: FloorPLan401 - FloorPlan430

    controller = ai2thor.controller.Controller()
    controller.start()
    controller.reset('FloorPlan28')
    # event = controller.step(dict(action='Initialize', gridSize=0.25,
    event = controller.step(dict(action='Initialize', gridSize=0.05,
                                     renderDepthImage=True,
                                     renderClassImage=True,
                                     renderObjectImage=True))

    # Numpy Array - shape (width, height, channels), channels are in RGB order
    print(event.frame)
    print(event.frame.shape)

    # to the mug
    # event = controller.step(dict(action='RotateRight'))
    # event = controller.step(dict(action='RotateRight'))
    # event = controller.step(dict(action='MoveAhead'))
    # event = controller.step(dict(action='MoveAhead'))
    # event = controller.step(dict(action='RotateRight'))
    # event = controller.step(dict(action='MoveAhead')) # two more here
    # event = controller.step(dict(action='RotateLeft'))
    # event = controller.step(dict(action='MoveAhead'))
    # event = controller.step(dict(action='MoveAhead'))
    # event = controller.step(dict(action='MoveAhead'))
    # event = controller.step(dict(action='MoveAhead'))

    # to the apple with gridSize=0.05
    event = controller.step(dict(action='RotateRight'))
    event = controller.step(dict(action='RotateRight'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='RotateRight'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='RotateLeft'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='LookDown'))
    all_object_names = [obj['name'] for obj in event.metadata['objects']]
    visible_objects = [obj for obj in event.metadata['objects'] if obj['visible']]
    # for i in range(50): event = controller.step(dict(action='MoveAhead'))

    import pdb;pdb.set_trace()

    # 98 75 126 109 - mug a bit to left middle centre
    # 165 75 194 109 - mug a bit to middle right bottom
    show_instance_segmentation(event, ['Mug', 'Apple'])
    show_bounding_boxes(event)
    show_bounding_boxes(event, ['mug', 'cup', 'Apple'], lines_between_boxes_and_crosshair=True)
    bool_list = check_if_focus_and_close_enough_to_object_type(event, 'Apple', verbose=True)
    # bool_list = check_if_focus_and_close_enough_to_object_type(event, 'Apple', verbose=True,
    #                                                            distance_threshold_3d=0.8)

    # # Show preprocessed image
    resolution = (128, 128)
    img = skimage.transform.resize(event.frame, resolution)
    plt.imshow(img) # show colour pre-processed (works in 0-1 range)
    plt.show()
    img = img.astype(np.float32)
    gray = rgb2gray(img)
    gray_unsqueezed = np.expand_dims(gray, 0)
    gray_3_channel = np.concatenate([gray_unsqueezed, gray_unsqueezed, gray_unsqueezed])
    gray_3_channel = np.moveaxis(gray_3_channel, 0, 2)
    plt.imshow(gray_3_channel)
    plt.show()

    # Can walk and step through environment interactively by copy/pasting commands and
    # deciding when to show bounding boxes. Can also check reward functions and have full control.
    import pdb;pdb.set_trace()
    # try pasting
    # event = controller.step(dict(action='MoveAhead'))
    # into the console
