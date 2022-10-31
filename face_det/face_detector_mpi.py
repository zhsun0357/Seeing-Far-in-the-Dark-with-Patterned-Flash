import cv2 as cv
import mediapipe as mp
import scipy.io as sio
import numpy as np
import glob
import natsort
import os
from pathlib import Path
import timeit
import pickle

def get_npdet(result):
    det_np = []
    for ii, rr in enumerate(result):
        if ii >=2:
            break
        bbox = np.asarray([rr.location_data.relative_bounding_box.xmin, \
                           rr.location_data.relative_bounding_box.ymin, \
                           rr.location_data.relative_bounding_box.width, \
                           rr.location_data.relative_bounding_box.height])
        kp = []
        for kk in rr.location_data.relative_keypoints:
            kp.append([kk.x, kk.y])
        kp = np.asarray(kp)
        det_np.append({'score': rr.score[0], 'bbox': bbox, 'kp': kp}) 
    return det_np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

results_folder = 'mediapipe_detection/'
Path(results_folder).mkdir(parents=True, exist_ok=True)

filenames = ['pf_face_det.png', 'uf_face_det.png', 'gt_face_det.png']
IMAGE_FILES = natsort.natsorted(filenames)
print(IMAGE_FILES)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.1) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv.imread(file)
        start = timeit.default_timer()
        try:
            results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        except:
            results = face_detection.process(image)
        stop = timeit.default_timer()
        print('Time: ', stop - start)

        # Draw face detections of each face.
        if not results.detections:
            print('No det')
            continue
        annotated_image = image.copy()
        
        ## save results in pickle format
        det_np = get_npdet(results.detections)
        with open(os.path.join(results_folder, 'det_result_' + file.replace('.png', '.pickle')), 'wb') as f:
            pickle.dump(det_np, f)

        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
        cv.imwrite(results_folder + 'annotated_image_' + file.replace('.png', '') + '.jpg', annotated_image)
