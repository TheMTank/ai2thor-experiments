import math
import random

import numpy as np
import torch
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
from matplotlib import pyplot as plt
import matplotlib.patches as patches


##############################
#------- Visualisation ------#
##############################

matplotlib_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def show_bounding_boxes(event, key_terms=None, lines_between_boxes_and_crosshair=False):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    show_all = True if not key_terms else False

    check_if = lambda x, name: True if x.lower() in name.lower() else False

    plt.imshow(event.frame, interpolation='nearest')
    for key, arr in event.instance_detections2D.items():
        if show_all or any([check_if(term, key) for term in key_terms]):
            x1, y1, x2, y2 = list(arr)
            bbox_color = matplotlib_colors[random.randint(0, len(matplotlib_colors) - 1)]
            print(key)
            print(x1, y1, x2, y2, ' with color: {}'.format(bbox_color))

            rect = patches.Rectangle((x1, y1), abs(x2 - x1), abs(y2 - y1), linewidth=1,
                                     edgecolor=bbox_color,
                                     facecolor='none')
            ax2.add_patch(rect)
            if lines_between_boxes_and_crosshair:
                crosshair_x, crosshair_y = 150, 150
                bbox_x_cent, bbox_y_cent = (x2 + x1) / 2, (y2 + y1) / 2
                x_coords, y_coords = [crosshair_x, bbox_x_cent], [crosshair_y, bbox_y_cent]
                plt.plot(x_coords, y_coords, marker='o', markersize=3, color=bbox_color)
                dist = math.sqrt((crosshair_x - bbox_x_cent) ** 2 +
                                 (crosshair_y - bbox_y_cent) ** 2)
                print('2D distance of bbox centre to crosshair: {}'.format(round(dist, 3)))
    plt.show()

def show_instance_segmentation(event, key_terms):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    show_all = True if not key_terms else False

    # check if objectType is within objectId
    check_if = lambda x, name: True if x.lower() in name.lower() else False

    event_frame_with_segmentation = event.frame.copy()
    for key, mask_arr in event.instance_masks.items():
        if show_all or any([check_if(term, key) for term in key_terms]):
            nonzero_indices = mask_arr.nonzero()

            event_frame_with_segmentation[nonzero_indices[0], nonzero_indices[1], :] = \
                np.array([255, 255, 255])  # only white for now

    plt.imshow(event_frame_with_segmentation, interpolation='nearest')
    plt.show()

##############################
#------- Preprocessing ------#
##############################

def get_word_to_idx(train_instructions):
    word_to_idx = {}
    for instruction_data in train_instructions:
        instruction = instruction_data
        for word in instruction.split(" "):
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    return word_to_idx

def turn_instruction_str_to_tensor(instruction, env):
    instruction_indices = []
    for word in instruction.split(" "):
        instruction_indices.append(env.task.word_to_idx[word])
    instruction_indices = np.array(instruction_indices)
    instruction_indices = torch.from_numpy(instruction_indices).view(1, -1)
    return instruction_indices

def unpack_state(state, env):
    """
    Returns (tensor image, None) if no natural language instructions and
    (tensor image, instruction embedding indices) if the task involves natural language.
    More specifically converts from numpy to tensor image and from string to embedding indices array
    and returns unpacked state into a tuple.
    """
    instruction_indices = None
    if not env.task.task_has_language_instructions:
        image_state = state
    else:
        # natural language instruction is within state so unpack tuple
        (image_state, instruction) = state

        instruction_indices = turn_instruction_str_to_tensor(instruction, env)
    image_state = torch.from_numpy(image_state)

    return image_state, instruction_indices

##############################
#----- Reward Functions -----#
##############################

def check_if_focus_and_close_enough_to_object_type(event, object_type='Mug',
                                                   distance_threshold_3d=1.0,
                                                   distance_threshold_2d=90, verbose=False):
    all_objects_for_object_type = [obj for obj in event.metadata['objects']
                                   if obj['objectType'] == object_type]

    if not all_objects_for_object_type:
        full_list_of_objects_types = [obj['objectType'] for obj in event.metadata['objects']]
        raise ValueError('Object type: "{}" which was chosen for object target from the last word '
                 'of instruction is not available in ai2thor at all or just is\'nt in this scene.'
                 ' The full list of object types in this scene: {}'.format(object_type,
                                                   ', '.join(full_list_of_objects_types)))

    bool_list = []
    for idx, obj in enumerate(all_objects_for_object_type):
        bounds = event.instance_detections2D.get(obj['objectId'])
        if bounds is None:
            continue

        x1, y1, x2, y2 = bounds
        euc_distance_to_obj = calculate_euc_distance_between_agent_and_object(
            event.metadata['agent'], obj)
        bool_list.append(check_if_focus_and_close_enough(x1, y1, x2, y2, euc_distance_to_obj,
                                                         distance_threshold_2d,
                                                         distance_threshold_3d, verbose))

    return sum(bool_list)

def calculate_euc_distance_between_agent_and_object(event_metadata_agent, obj, two_dimensions=True):
    a_x, a_y, a_z = event_metadata_agent['position']['x'], \
                    event_metadata_agent['position']['y'], \
                    event_metadata_agent['position']['z']
    obj_x, obj_y, obj_z = obj['position']['x'], obj['position']['y'], obj['position']['z']
    if two_dimensions:
        euclidean_distance_to_obj = math.sqrt((obj_x - a_x) ** 2 + (obj_z - a_z) ** 2)
    else:
        euclidean_distance_to_obj = math.sqrt((obj_x - a_x) ** 2 + (obj_y - a_y) ** 2 +
                                              (obj_z - a_z) ** 2)
    return euclidean_distance_to_obj

def check_if_focus_and_close_enough(x1, y1, x2, y2, distance_3d, distance_threshold_2d,
                                    distance_threshold_3d, verbose=False):
    focus_bool = is_bounding_box_centre_close_to_crosshair(x1, y1, x2, y2, verbose=verbose,
                                                       distance_threshold_2d=distance_threshold_2d)
    close_bool = euclidean_close_enough_3d(distance_3d, distance_threshold_3d, verbose=verbose)

    if verbose:
        print('Object within 2D distance of crosshair: {}. Object close enough with 3D distance: {}'.format(focus_bool, close_bool))

    return True if focus_bool and close_bool else False

def is_bounding_box_centre_close_to_crosshair(x1, y1, x2, y2, distance_threshold_2d, verbose=False):
    """
    object's bounding box has to be mostly within the 100x100 middle of the image.
    Could also use distance within obj type but decided to do this instead
    """
    bbox_x_cent, bbox_y_cent = (x2 + x1) / 2, (y2 + y1) / 2
    dist = math.sqrt((150 - bbox_x_cent) ** 2 + (150 - bbox_y_cent) ** 2)
    if verbose:
        print('Euclidean 2D distance to crosshair: {}. distance_threshold_2d: {}'.format(dist,
                                                                            distance_threshold_2d))
    return True if dist < distance_threshold_2d else False

def euclidean_close_enough_3d(distance, distance_threshold_3d, verbose=False):
    if verbose:
        print('Euclidean 3D distance: {}. distance_threshold_3d: {}'.format(distance,
                                                                    distance_threshold_3d))
    return True if distance < distance_threshold_3d else False
