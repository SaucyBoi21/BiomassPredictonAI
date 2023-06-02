#############################
#   Step 1: 





##############################

from threading import Condition
import time
from time import sleep
import numpy as np
import os
import cv2

from seekcamera import (
    SeekCameraIOType,
    SeekCameraColorPalette,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCamera,
    SeekFrame,
)

path = '/home/jsc95/thermal_imaging'
timestamp = time.strftime("%b-%d-%Y_%H%M")

class Renderer:
    """Contains camera and image data required to render images to the screen."""

    def __init__(self):
        self.busy = False
        self.frame = SeekFrame()
        self.camera = SeekCamera()
        self.frame_condition = Condition()
        self.first_frame = True


def on_frame_vis(_camera, camera_frame, renderer):
    """Async callback fired whenever a new frame is available.

    Parameters
    ----------
    _camera: SeekCamera
        Reference to the camera for which the new frame is available.
    camera_frame: SeekCameraFrame
        Reference to the class encapsulating the new frame (potentially
        in multiple formats).
    renderer: Renderer
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the renderer object.
    """

    # Acquire the condition variable and notify the main thread
    # that a new frame is ready to render. This is required since
    # all rendering done by OpenCV needs to happen on the main thread.
    with renderer.frame_condition:
        renderer.frame = camera_frame.color_argb8888
        renderer.frame_condition.notify()


def on_event_vis(camera, event_type, event_status, renderer):
    """Async callback fired whenever a camera event occurs.

    Parameters
    ----------
    camera: SeekCamera
        Reference to the camera on which an event occurred.
    event_type: SeekCameraManagerEvent
        Enumerated type indicating the type of event that occurred.
    event_status: Optional[SeekCameraError]
        Optional exception type. It will be a non-None derived instance of
        SeekCameraError if the event_type is SeekCameraManagerEvent.ERROR.
    renderer: Renderer
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the Renderer object.
    """
    print("{}: {}".format(str(event_type), camera.chipid))

    if event_type == SeekCameraManagerEvent.CONNECT:
        if renderer.busy:
            return

        # Claim the renderer.
        # This is required in case of multiple cameras.
        renderer.busy = True
        renderer.camera = camera

        # Indicate the first frame has not come in yet.
        # This is required to properly resize the rendering window.
        renderer.first_frame = True

        # Set a custom color palette.
        # Other options can set in a similar fashion.
        camera.color_palette = SeekCameraColorPalette.TYRIAN

        # Start imaging and provide a custom callback to be called
        # every time a new frame is received.
        camera.register_frame_available_callback(on_frame_vis, renderer)
        camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)

    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        # Check that the camera disconnecting is one actually associated with
        # the renderer. This is required in case of multiple cameras.
        if renderer.camera == camera:
            # Stop imaging and reset all the renderer state.
            camera.capture_session_stop()
            renderer.camera = None
            renderer.frame = None
            renderer.busy = False

    elif event_type == SeekCameraManagerEvent.ERROR:
        print("{}: {}".format(str(event_status), camera.chipid))

    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return
    
    
def on_frame_val(camera, camera_frame, file):
    """Async callback fired whenever a new frame is available.

    Parameters
    ----------
    camera: SeekCamera
        Reference to the camera for which the new frame is available.
    camera_frame: SeekCameraFrame
        Reference to the class encapsulating the new frame (potentially
        in multiple formats).
    file: TextIOWrapper
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the open CSV file to which
        to log data.
    """
    frame = camera_frame.thermography_float

    print(
        "frame available: {cid} (size: {w}x{h})".format(
            cid=camera.chipid, w=frame.width, h=frame.height
        )
    )

    # Append the frame to the CSV file.
    np.savetxt(file, frame.data, fmt="%.1f")


def on_event_val(camera, event_type, event_status, _user_data):

    print("{}: {}".format(str(event_type), camera.chipid))
    
    if event_type == SeekCameraManagerEvent.CONNECT:
    # Open a new CSV file with the unique camera chip ID embedded.
        try:
            file = open("thermography-" + camera.chipid + timestamp +".csv", "w")
        except OSError as e:
            print("Failed to open file: %s" % str(e))
            return
    
    # Start streaming data and provide a custom callback to be called
    # every time a new frame is received.
        camera.register_frame_available_callback(on_frame_val, file)
        camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)
        sleep(2.2)
        #Stop the camera session after a single frame is captured. 
        camera.capture_session_stop()
    
    
    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        pass
    
    elif event_type == SeekCameraManagerEvent.ERROR:
        print("{}: {}".format(str(event_status), camera.chipid))
  
    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return


def main():
    window_name = "Seek Thermal - Python OpenCV Sample"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create a context structure responsible for managing all connected USB cameras.
    # Cameras with other IO types can be managed by using a bitwise or of the
    # SeekCameraIOType enum cases.
    with SeekCameraManager(SeekCameraIOType.USB) as manager:
        # Start listening for events.
        renderer = Renderer()
        manager.register_event_callback(on_event_vis, renderer)

        while True:
            # Wait a maximum of 150ms for each frame to be received.
            # A condition variable is used to synchronize the access to the renderer;
            # it will be notified by the user defined frame available callback thread.
            with renderer.frame_condition:
                if renderer.frame_condition.wait(150.0 / 1000.0):
                    img = renderer.frame.data
                    

                    # Resize the rendering window.
                    if renderer.first_frame:
                        (height, width, _) = img.shape
                        cv2.resizeWindow(window_name, width * 2, height * 2)
                        renderer.first_frame = False

                    # Render the image to the window.
                    cv2.imshow(window_name, img)
                    

            # Process key events.
            cv2.waitKey(1000)
            #if key == ord("q"):
            #    break
            #if key == ord("c"):
            capture = True
            cv2.imwrite(os.path.join(path, timestamp + "_thermal.jpg"), img)
            #    break
            # Check if the window has been closed manually.
            

    cv2.destroyWindow(window_name)
    
    with SeekCameraManager(SeekCameraIOType.USB) as manager:
        # Start listening for events.
        manager.register_event_callback(on_event_val)
        while True:
            sleep(10)
            break
            
            


if __name__ == "__main__":
    main()
