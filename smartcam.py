import numpy as np
import cv2
import sys # for getting command line arguments


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # The HOG detector returns slightly larger rectangles than the real objects.
        # So, gotta slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


# Stores the command line argument.
# Should say either test or deploy.
# If it is not one of these, then show the startup message.
if len(sys.argv) == 1:
    runmode = "no args given"
else:
    runmode = sys.argv[1]

# Zero means use the first available camera. If using a USB webcam, value should be changed to 1
camnumber = 0


# Set up the Histogram Oriented Graph
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# How many people were detected in the previous frame
# This is to avoid multiple images of the same person
last_frame_found = 0


# This variable is just to index the pictures' filenames cleanly
frame_index = 0


if runmode == "deploy":
    # Open the webcam
    vidcap = cv2.VideoCapture(camnumber)

    while True:
        # Capture frame
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
        found_filtered = []

        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)

        draw_detections(frame, found)
        draw_detections(frame, found_filtered, 3)


        # Uncomment the following line for a frame-by-frame output of how many people were seen
        # print('%d (%d) found' % (len(found_filtered), len(found)))


        # Display the resulting frame
        cv2.imshow('frame', frame)

        # If there are more people in this frame than there were in the last frame, save the image
        if len(found_filtered) > last_frame_found:
            print "Gotta save this frame!"
            frame_name = "person_" + str(frame_index) + ".jpg"
            cv2.imwrite(frame_name, frame)
            frame_index += 1
        else:
            print "No save"
        # Finally, update the value of last_frame_found
        last_frame_found = len(found)

        # Quit with the q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    vidcap.release()
    cv2.destroyAllWindows()


elif runmode == "test":
    # Open the webcam
    vidcap = cv2.VideoCapture(camnumber)

    while True:
        # Capture frame
        ret, frame = vidcap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)

        draw_detections(frame, found)
        draw_detections(frame, found_filtered, 3)


        # Uncomment the following line for a frame-by-frame output of how many people were seen
        # print('%d (%d) found' % (len(found_filtered), len(found)))


        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vidcap.release()
    cv2.destroyAllWindows()

else:
    print "Hi! Welcome to SmartCam: my Computer Vision Project"
    print "Please run SmartCam with the argument \"test\" for testing, setup or demonstration."
    print "To deploy SmartCam fully, and to save images, use the argument \"deploy\""
    print "For feedback or suppport, contact me at tjsj78@gmail.com"
