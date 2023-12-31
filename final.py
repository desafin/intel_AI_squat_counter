from decoder import OpenPoseDecoder
from openvino.tools import mo
import collections
import time
from pathlib import Path
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from openvino.runtime import Core
import psutil

# 모델 불러오기

model_ir = mo.convert_model(
    "u2net.onnx",
    mean_values=[123.675, 116.28 , 103.53],
    scale_values=[58.395, 57.12 , 57.375],
    compress_to_fp16=True
)



ie_core = Core()
compiled_model_ir = ie_core.compile_model(model=model_ir, device_name="GPU.0")


# Get the names of input and output layers.
input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)










# A directory where the model will be downloaded.
base_model_dir = Path("model")

# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"

# Selected precision (FP32, FP16, FP16-INT8).
precision = "FP16-INT8"

model_path = "human-pose-estimation-0001.xml"

# Read the network from a file.
model = ie_core.read_model(model_path)
# Let the AUTO device decide where to load the model (you can use CPU, GPU or MYRIAD as well).
compiled_model = ie_core.compile_model(model=model, device_name="GPU.0", config={"PERFORMANCE_HINT": "LATENCY"})

# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# Get the input size.
height, width = list(input_layer.shape)[2:]



decoder = OpenPoseDecoder()


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)




def remove_background(frame, compiled_model_ir, output_layer_ir):#배경제거
    # Preprocess the frame
    resized_frame = cv2.resize(src=frame, dsize=(512, 512))
    input_frame = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
    bg_removed_result = compiled_model_ir([input_frame])[output_layer_ir]

    # Resize the network result to the frame shape and round the values
    resized_result = np.rint(cv2.resize(src=np.squeeze(bg_removed_result), dsize=(frame.shape[1], frame.shape[0]))).astype(np.uint8)

    # Create a copy of the frame and set all background values to 255 (white)
    bg_removed_result = frame.copy()
    bg_removed_result[resized_result == 0] = 255

    return bg_removed_result







# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):

    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )



    kernel_size = (kernel_size, kernel_size)


    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )


    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)


    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results(img, pafs, heatmaps):
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()




    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores





colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))





default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))










global shoulder
shoulder =0

global foot
foot=0

global distance
distance=0

global count
count=0

global isdown
isdown=False

global userheight
userheight =0




def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    global shoulder
    global foot
    global distance
    global count
    global isdown
    global userheight


    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
                text = f'({p[0]}, {p[1]})'
                cv2.putText(img, text, tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)





                if i==5:#왼쪾어께
                    shoulder=f'{p[1]}'
                    #print(shoulder)

                if i==15:#왼쪽발
                    foot = f'{p[1]}'
                    #print(foot)

                distance=abs(int(foot)-int(shoulder))
                #print(distance)






        if distance<(userheight*0.8):

            if isdown==False:
                count=count+1
                isdown=True

        else:
            isdown = False




       # Draw limbs.-000
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)

    return img


def run_pose_estimation_webcam(flip=False, use_popup=False, skip_first_frames=0):
    global count
    global userheight



    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")


    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")



    processing_times = collections.deque()

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()

        frame = cv2.resize(frame, None, fx=2, fy=2)
        cv2.namedWindow("HELL quat", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        if not ret:
            print("Failed to read the frame")
            break



        # If the frame is larger than full HD, reduce size to improve performance.
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Resize the image and change dimensions to fit neural network input.ee
        input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

        # Measure processing time.
        start_time = time.time()
        # Get results.
        results = compiled_model([input_img])
        stop_time = time.time()

        pafs = results[pafs_output_key]
        heatmaps = results[heatmaps_output_key]
        # Get poses from network results.
        poses, scores = process_results(frame, pafs, heatmaps)

        frame = remove_background(frame, compiled_model_ir, output_layer_ir)  # 백그라운드 제거 함수
        frame = draw_poses(frame, poses, 0.1)






        processing_times.append(stop_time - start_time)
        # Use processing times from last 200 frames.
        if len(processing_times) > 200:
            processing_times.popleft()

        _, f_width = frame.shape[:2]
        # Mean processing time [ms]
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (0,900),cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)




        text=str(f"Count: {count}")
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if use_popup:
            if userheight == 0:
                text = "please press ""e"" set user's height"
                cv2.putText(frame, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif userheight != 0:
                text = "user's height is "+str(userheight)
                cv2.putText(frame, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

            cv2.imshow("HELL quat", frame)
            #print(str(psutil.cpu_percent())+"%")


            key = cv2.waitKey(40)

            if key == 27:# Escape key (27)
                print("exit program")
                break
            if key == ord('w'):
                print("count to =0")
                count=0



            if key == ord('e'):#e를 누르면 사용자의 신장이 지정됨
                userheight=distance
                print("신장은="+str(userheight))
                print(str(userheight*0.8)+"만큼 숙여주세요")



    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

run_pose_estimation_webcam(flip=True, use_popup=True)




