# import the necessary packages

import argparse
import cv2 as cv
from object_detection import object_detector
from sort import *
from timeit import default_timer as timer

import torch


def get_peak_points(heatmaps):
    N, C, H, W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy, xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            score = heatmaps[i, j].max()
            peak_points.append([x, y, score])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def predict_pose(image, model):
    if image is None:
        return
    image = cv.resize(image, (128, 128))
    image = image/255
    image = image.transpose([2, 0, 1])
    input_tensor = torch.from_numpy(image).float().unsqueeze(0)
    output = model.forward(input_tensor.to(torch.device('cuda:0')))[0].cpu().data.numpy()[np.newaxis, ...]

    N, C, H, W = output.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy, xx = np.where(output[i, j] == output[i, j].max())
            y = yy[0]
            x = xx[0]
            score = output[i, j].max()
            peak_points.append([x, y, score])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)[0]

    return all_peak_points


def process(args):

    tracker = Sort(max_age=9, min_hits=3)
    memory = {}

    stream = cv.VideoCapture(args.input if args.input else 0)

    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
    else:
        classes = list(np.arange(0, 100))

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    writer = None
    frameIndex = 0
    # loop over frames from the video file stream
    prev_time = timer()
    accum_time = 0
    curr_fps = 0
    predictor = object_detector(args.model, args.config)
    pose_model = torch.jit.load('KFS_NET.pt')

    fps = ""
    while stream.isOpened():
        total_time1 = timer()
        boxes = []
        confidences = []
        classIDs = []
        midPoint = []
        getimg_time1 = timer()
        # read the next frame from the file
        (grabbed, frame) = stream.read()
        
        if frame is None:
            break
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        getimg_time2 = timer()
        print("Read Time:", (getimg_time2 - getimg_time1)*1000, "ms")
        det_time1 = timer()

        predictions = predictor.predict(frame)

        for output in predictions:
            for detection in output:
                scores = detection[5:]  
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args.thr:

                    box = detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                    (centerX, centerY, width, height) = box.astype("int") 
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    center = [centerX, centerY]
                    midPoint.append(center)


        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.4)

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])

        det_time2 = timer()
        print("Detect Time:", (det_time2-det_time1)*1000, "ms")  

        track_time1 = timer()        
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
        track_time2 = timer()   
        print("Track Time:", (track_time2 - track_time1) * 1000, "ms")    

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                label = classes[classID]
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv.rectangle(frame, (x, y), (w, h), color, 2)

                # 处理ROI
                y = 0 if y < 0 else int(y*0.9)
                x = 0 if x < 0 else int(x*0.9)
                h = frame.shape[0]-y if h*1.1 > frame.shape[0] else int(h*1.1)
                w = frame.shape[1]-x if h*1.1 > frame.shape[1] else int(w*1.1)
                roi = frame[y: h, x: w]
                # predict keypoints
                pose_time1 = timer()
                pose = predict_pose(roi, pose_model)
                pose_time2 = timer()
                print("Pose Time:", (pose_time2 - pose_time1)*1000, "ms")

                # draw points
                for pt_index in range(0, pose.shape[0]):
                    if pose[pt_index, 2] >= threshold:
                        p = (x + int(pose[pt_index, 0] * roi.shape[1]/128), y + int(pose[pt_index, 1] * roi.shape[0]/128))
                        cv.circle(frame, p, 3, (100, 100, 200), 4)

                # draw lines
                for part_index in range(0, len(BODY_PARTS_KPT_IDS)):
                    if pose[BODY_PARTS_KPT_IDS[part_index][0], 2] >= threshold and pose[BODY_PARTS_KPT_IDS[part_index][1], 2] >= threshold:
                        p1 = (x + int(pose[BODY_PARTS_KPT_IDS[part_index][0], 0] * roi.shape[1]/128),
                              y + int(pose[BODY_PARTS_KPT_IDS[part_index][0], 1] * roi.shape[0]/128))
                        p2 = (x + int(pose[BODY_PARTS_KPT_IDS[part_index][1], 0] * roi.shape[1]/128),
                              y + int(pose[BODY_PARTS_KPT_IDS[part_index][1], 1] * roi.shape[0]/128))

                        cv.line(frame, p1, p2, (200, 100, 100), 2, 8)


                # 轨迹线
                # if indexIDs[i] in previous:
                #     previous_box = previous[indexIDs[i]]
                #     (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                #     (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                #     p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                #     p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                #     # print(previous_box)
                #     # draw lines
                #     cv.line(frame, p0, p1, color, 3)

                text = "{}_{}".format(label, indexIDs[i])

                cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        cv.putText(frame, text=fps, org=(3, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.8, color=(255, 0, 0), thickness=1)

        total_time2 = timer()
        print("Total Time:", (total_time2 - total_time1) * 1000, "ms")

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv.VideoWriter_fourcc(*"XVID")
            video_fps = stream.get(cv.CAP_PROP_FPS)
            cv.putText(frame, 'fps: %d' %(video_fps), (9, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5)
            # writer = cv.VideoWriter(args.output, fourcc, video_fps, (frame.shape[1], frame.shape[0]), True)
            writer = cv.VideoWriter(args.output, fourcc, 12, (frame.shape[1], frame.shape[0]), True)
        # write the output frame to disk
        writer.write(frame)

        cv.namedWindow("hasil", cv.WINDOW_NORMAL)
        cv.resizeWindow("hasil", 640, 480)
        cv.imshow('hasil', frame)
        cv.waitKey(1)

        # increase frame index
        frameIndex += 1

    # release the file pointers
    print("[INFO] cleaning up...")
    cv.destroyAllWindows()
    writer.release()
    stream.release()


if __name__ == '__main__':
    BODY_PARTS_KPT_IDS = [[0, 1], [0, 3], [1, 2], [3, 4], [0, 5], [0, 11], [5, 6], [6, 7],
                          [11, 12], [12, 13], [5, 17], [17, 8], [5, 18], [18, 8], [11, 17],
                          [17, 14], [11, 18], [18, 14], [8, 9], [9, 10], [14, 15], [15, 16], [8, 19], [14, 19]]
    threshold = 0.2
    parser = argparse.ArgumentParser(description='Object Detection and Tracking on Video Streams')

    parser.add_argument('--input', default='input/1.mp4', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    parser.add_argument('--output', default='result.mp4', help='Path to save output as video file. If nothing is given, the output will not be saved.')

    parser.add_argument('--model', default='models/yolov4_best.weights',
                        help='Path to a binary file of model contains trained weights. '
                             'It could be a file with extensions .weights (Darknet)')

    parser.add_argument('--config', default='models/yolov4.cfg',
                        help='Path to a text file of model contains network configuration. '
                             'It could be a file with extensions .cfg (Darknet)')

    parser.add_argument('--classes', default='models/pig.names', help='Optional path to a text file with names of classes to label detected objects.')

    parser.add_argument('--thr', type=float, default=0.02, help='Confidence threshold for detection')

    args = parser.parse_args()

    process(args)
