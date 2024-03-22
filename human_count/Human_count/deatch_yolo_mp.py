import os
import time
import yaml
import argparse
import numpy as np
from onnxruntime import InferenceSession
from deploy.third_engine.onnx.preprocess import Compose
from deploy.third_engine.onnx.infer import get_test_images,PredictConfig
from PIL import Image, ImageDraw
from ppdet.modeling.mot.utils import Detection, get_crops, scale_coords, clip_box
from ppdet.modeling.mot.tracker.jde_tracker import JDETracker
import tqdm
import copy
import cv2
from multiprocessing import Process, RawArray, Lock
import multiprocessing as mp

def trans_cv2(img,trans):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv_t = trans.transforms
    im_info = {
        "im_shape": np.array(
            img.shape[:2], dtype=np.float32),
        "scale_factor": np.array(
            [1., 1.], dtype=np.float32)
    }
    for t in cv_t:
        img, im_info = t(img, im_info)
    inputs = copy.deepcopy(im_info)
    inputs['image'] = img
    return inputs

def img2memory(lock, img, raws): # set and lock memory
    h, w, _ = img.shape
    # print("share shape",img.shape)
    lock.acquire()  # 获取线程锁，确保在修改共享内存时只有一个线程访问 # todo
    memoryview(raws).cast('B')[:img.size] = img.ravel()  # 将图像数据写入共享内存的对应位置
    lock.release()  # 释放线程锁，允许其他线程访问共享内存
    return w, h

def read_camera(lock, raws,v_root,gb_var):
    # address = 'rtsp://{}@{}:554/h264/ch1/main/av_stream'  # 创建 RTSP 地址
    cap = cv2.VideoCapture(v_root)

    ret, frame = cap.read()
    # pcw, pch, _ = [1440, 800, 3]
    pch, pcw,_ = frame.shape   # # 720,1280,3
      # 打开 RTSP 地址，创建 VideoCapture 对象
    gap_frame = 2
    count = -1
    tem_fr = -1
    while  cap.isOpened():  # 在 flag 为真且时间限制内的循环中读取帧数据
        if gb_var['fr'] >tem_fr: # ckpt : videos

            ret, img = cap.read()
            count += 1
            if count % gap_frame >0  or count <2000:
                continue
            if frame is None:
                print("empty img")
                break
            if not ret: continue  # 如果没有成功读取到帧，则跳过当前循环
            img2memory(lock, img, raws)  # 调用 img2memory 函数，将帧数据写入共享内存
            tem_fr = gb_var['fr']

def run_x(raw_pth,rations,rshape,infer_config,gb_var):
    #
    tracker = JDETracker()
    tracker.track_buffer, tracker.conf_thres = 30, 0.75  # todo parse
    transforms = Compose(infer_config.preprocess_infos)
    device = 'cuda'
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
    predictor = InferenceSession("/home/cata/Desktop/PaddleDetection/pdmod/bytetrack_yolox/yolox.onnx", providers=providers)
    save_idx = 0
    pass_threshold = 0.8
    mini_area = 20  # todo pares
    # total_pass = 0
    id_info = {}
    counts_dict_ = {"inline": 0, "outline": 0}
    total_start = time.time()
    count = -1
    ch,cw,_ = rshape
    red_line = [(cw * rations[0], ch * rations[1]), (cw * rations[2], ch * rations[3])]
    in_id = []
    out_id = []
    total_in = 0
    total_out = 0
    while 1:
        frame = np.frombuffer(raw_pth, dtype=np.uint8).reshape(rshape)
        start_time = time.time()
        inputs = trans_cv2(frame, transforms)
        # cw,ch = inputs
        inputs_name = [var.name for var in predictor.get_inputs()]
        inputs = {k: inputs[k][None,] for k in inputs_name}
        outputs = predictor.run(output_names=None, input_feed=inputs)
        tem_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        bboxes = np.array(outputs[0])
        outs = {}
        outs["bbox"] = bboxes
        outs["bbox_num"] = np.array([100])

        if len(outs['bbox']) > 0:
            # detector outputs: pred_cls_ids, pred_scores, pred_bboxes
            pred_cls_ids = outs['bbox'][:, 0:1]
            pred_scores = outs['bbox'][:, 1:2]
            pred_bboxes = outs['bbox'][:, 2:]
            pred_dets_old = np.concatenate(
                (pred_cls_ids, pred_scores, pred_bboxes), axis=1)
        ori_image_shape = tem_im.size
        pred_xyxys, keep_idx = clip_box(pred_bboxes, ori_image_shape)
        pred_cls_ids = pred_cls_ids[keep_idx[0]]
        pred_scores = pred_scores[keep_idx[0]]
        pred_dets = np.concatenate(
            (pred_cls_ids, pred_scores, pred_xyxys), axis=1)
        pred_embs = None
        online_targets_dict = tracker.update(pred_dets_old, pred_embs)
        draw = ImageDraw.Draw(tem_im)
        tem_ids = online_targets_dict[0]  # class:0
        tem_in = 0
        tem_out = 0
        for t in tem_ids:
            tlwh = t.tlwh
            tscore = t.score
            tid = t.track_id
            if t.score < pass_threshold: continue
            if tlwh[2] * tlwh[3] < mini_area: continue
            if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                3] > tracker.vertical_ratio:
                continue
            mid_c = (tlwh[0] + 1.0 * tlwh[2], tlwh[1] + 0.5 * tlwh[3])
            bx = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
            draw.rectangle(bx, outline='red', )
            draw.point(mid_c, fill="yellow")  # +0.5*tlwh[3]
            # draw.text((bx[0], bx[1]), text=str(tid), fill='blue', )
            # pass_point = abs((mid_c[0] / red_line[1][0])) * (red_line[1][1] - red_line[0][1]) + red_line[0][1]
            pass_point = (mid_c[0] - red_line[0][0]) / (red_line[1][0] - red_line[0][0]) * (
                        red_line[1][1] - red_line[0][1]) + red_line[0][1]
            if mid_c[0] < red_line[0][0]:
                continue
            if tid in id_info.keys():
                tem_state = id_info[tid]
            if tid not in id_info.keys():
                tem_state = 0
                id_info[tid] = 0 # todo unsure
            if mid_c[1] - pass_point < 0:
                id_info[tid] = -1
            if mid_c[1] - pass_point >= 0+6:
                id_info[tid] = 1

            if tem_state * id_info[tid] == -1 and id_info[tid] == -1:
                if tid not in out_id:
                    out_id.append(tid)
                    tem_out += 1
            if tem_state * id_info[tid] == -1 and id_info[tid] == 1:
                if tid not in in_id:
                    in_id.append(tid)
                    tem_in += 1

        total_in += tem_in
        total_out += tem_out
        if tem_in > 0 or tem_out>0:
            draw.line(red_line, fill="green", width=3)
        else:
            draw.line(red_line, fill="red", width=3)
        fps = 1 / (time.time() - start_time)
        print("fps",fps)
        color1 = (138, 43, 226)
        im_cv = cv2.cvtColor(np.array(tem_im), cv2.COLOR_RGB2BGR)
        image_1 = cv2.putText(im_cv, "out " + str(total_out), (int(cw * 0.48), 50), 0, 1, color1, 3)
        image_1 = cv2.putText(im_cv, "in " + str(total_in), (int(cw * 0.48), 25), 0, 1, color1, 3)
        cv2.imshow('image', image_1)
        cv2.waitKey(1)
        gb_var['fr']  +=1
        # tem_im.save(save_path+str(save_idx)+".png")

    print("total times:", time.time() - total_start)
    cap.release()
    cv2.destroyAllWindows()

def predict_video(infer_config, img_list,rations=[0.55,0.35,1,0.42]):  # [0.5,0.32,1,0.40]
    # load preprocess transforms
    tem_cap = cv2.VideoCapture(v_root)
    ret, img = tem_cap.read()
    ch,cw,_= img.shape
    tem_cap.release()
    Image_raw = RawArray('B', cw * ch * 3)
    start_rat = 0
    lock = Lock()
    m = mp.Manager()
    gb_var = m.dict()
    gb_var['fr'] = 0
    # shared_var = mp.Manager().value("i",0)

    process1 = Process(target=run_x, args=(Image_raw,rations,img.shape,infer_config,gb_var))  # (raw_pth,rations,rshape,predictor)
    process1.start()
    process_main = Process(target=read_camera, args=(lock, Image_raw,v_root,gb_var))
    process_main.start()
    process_main.join()
    process1.join()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--infer_cfg", type=str, default="/home/cata/Desktop/PaddleDetection/pdmod/bytetrack_yolox/infer_cfg.yml")
parser.add_argument(
    '--onnx_file', type=str, default="/home/cata/Desktop/PaddleDetection/pdmod/bytetrack_yolox/yolox.onnx", help="onnx model file path",)
parser.add_argument("--image_dir", type=str)
parser.add_argument("--image_file", type=str,default= "/home/cata/Desktop/PaddleDetection/demo/ped.png")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    FLAGS = parser.parse_args()
    # load image list
    img_list = os.listdir("/home/cata/Desktop/PaddleDetection/videos/town/town/")
    sl = len(img_list[0]) - len(".png")
    sl = 8
    img_lst = [str(0)*(sl-len(str(i+1)))+str(i+1)+".png"  for i in range(len(img_list))]
    # load predictor
    fr = 0
    # load infer config
    infer_config = PredictConfig(FLAGS.infer_cfg)
    # predict_image(infer_config, predictor, img_lst)
    v_root = "/home/cata/Desktop/PaddleDetection/videos/4.3pm.mp4"
    predict_video(infer_config, v_root)



