import threading
import time

import ai2thor.controller


def run_simple_example():
    """
    Taken from here: http://ai2thor.allenai.org/tutorials/examples
    """
    controller = ai2thor.controller.Controller()
    controller.start()

    # Kitchens: FloorPlan1 - FloorPlan30
    # Living rooms: FloorPlan201 - FloorPlan230
    # Bedrooms: FloorPlan301 - FloorPlan330
    # Bathrooms: FloorPLan401 - FloorPlan430

    controller.reset('FloorPlan28')
    controller.step(dict(action='Initialize', gridSize=0.25))

    event = controller.step(dict(action='MoveAhead'))

    # Numpy Array - shape (width, height, channels), channels are in RGB order
    event.frame

    # Numpy Array in BGR order suitable for use with OpenCV
    event.cv2img

    # current metadata dictionary that includes the state of the scene
    event.metadata

    controller.stop()

def run_calling_complex_actions():
    """
    Examples of how to interact with environment internals e.g. picking up, placing and
    opening objects.
    Taken from here: http://ai2thor.allenai.org/tutorials/examples
    """
    controller = ai2thor.controller.Controller()
    controller.start()

    controller.reset('FloorPlan28')
    controller.step(dict(action='Initialize', gridSize=0.25))

    controller.step(dict(action='Teleport', x=-1.25, y=1.00, z=-1.5))
    controller.step(dict(action='LookDown'))
    event = controller.step(dict(action='Rotate', rotation=90))
    # In FloorPlan28, the agent should now be looking at a mug
    for obj in event.metadata['objects']:
        if obj['visible'] and obj['pickupable'] and obj['objectType'] == 'Mug':
            event = controller.step(dict(action='PickupObject', objectId=obj['objectId']),
                                    raise_for_failure=True)
            mug_object_id = obj['objectId']
            break

    # the agent now has the Mug in its inventory
    # to put it into the Microwave, we need to open the microwave first

    event = controller.step(dict(action='LookUp'))
    for obj in event.metadata['objects']:
        if obj['visible'] and obj['openable'] and obj['objectType'] == 'Microwave':
            event = controller.step(dict(action='OpenObject', objectId=obj['objectId']),
                                    raise_for_failure=True)
            receptacle_object_id = obj['objectId']
            break

    event = controller.step(dict(action='MoveRight'), raise_for_failure=True)
    event = controller.step(dict(action='PutObject',
                                 receptacleObjectId=receptacle_object_id,
                                 objectId=mug_object_id),
                            raise_for_failure=True)

    # close the microwave
    event = controller.step(dict(
        action='CloseObject',
        objectId=receptacle_object_id), raise_for_failure=True)

    controller.stop()

def run_multithreaded():
    """
    Stress test and also shows how multi-threading can be used to greatly speed up processing,
    specially to support the rendering of class, object and depth images.
    Adapted from here: http://ai2thor.allenai.org/tutorials/examples

    Extra analysis done on adding unity information. Important for training models to know.
    ~67 FPS with 1 thread no extra info
    ~61 FPS with 1 thread added class info
    ~18 FPS with 1 thread added Object info on top
    ~17 FPS with 1 thread added Depth info on top

    ~70 FPS with 2 threads and no depth, class and object image
    ~15 FPS with 2 threads and all three of those

    Good examples of how to multi-thread are below
    """
    thread_count = 3

    def run(thread_num):
        """
        Runs 5 iterations of 10 steps of the environment with the different rendering options
        :param thread_num: (int) Number of threads to launch
        """
        env = ai2thor.controller.Controller()
        env.start()

        render_depth_image, render_class_image, render_object_image = False, False, False

        for i in range(5):
            t_start = time.time()
            env.reset('FloorPlan1')
            env.step({'action': 'Initialize', 'gridSize': 0.25})

            # Compare the performance with all the extra added information
            # Big take away is that Object instance information makes it much slower
            if i == 2:
                render_class_image = True
                print('Thread num: {}. Added Class info'.format(thread_num))
            elif i == 3:
                render_object_image = True
                print('Thread num: {}. Added Object info'.format(thread_num))
            elif i == 4:
                render_depth_image = True
                print('Thread num: {}. Added Depth info'.format(thread_num))

            env.step(dict(action='Initialize',
                          gridSize=0.25,
                          renderDepthImage=render_depth_image,
                          renderClassImage=render_class_image,
                          renderObjectImage=render_object_image))
            print('Thread num: {}. init time: {}'.format(thread_num, time.time() - t_start))
            t_start_total = time.time()
            num_steps = 50
            for _ in range(num_steps):
                env.step({'action': 'MoveAhead'})
            total_time = time.time() - t_start_total
            print('Thread num: {}. Total time for 10 steps: {}. {:.2f} fps'.
                  format(thread_num, total_time, num_steps / total_time))
        env.stop()

    threads = [threading.Thread(target=run, args=(thread_num, ))
               for thread_num in range(thread_count)]
    for thread in threads:
        thread.daemon = True
        thread.start()
        time.sleep(1)

    for thread in threads:
        # calling join() in a loop/timeout to allow for Python 2.7
        # to be interrupted with SIGINT
        while thread.isAlive():
            thread.join(1)

    print('done')

if __name__ == '__main__':
    run_simple_example()
    run_calling_complex_actions()
    run_multithreaded()
