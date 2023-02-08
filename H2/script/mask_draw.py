import os, sys, math, cv2, mmcv, numpy
from mmdet.apis import inference_detector, init_detector

def mask_draw(src_file, dst_file, mod_file, conf_file, threshold, device):
    model = init_detector(conf_file, mod_file, device=device)
    score_thr = threshold
    frames = mmcv.VideoReader(src_file)
    fourcc = "XVID"
    fps = math.ceil(frames.fps)
    resolution = (frames.width, frames.height)
    vwriter = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*fourcc), fps, resolution)
    for fid in mmcv.track_iter_progress(range(len(frames))):
        if frames[fid] is None:
            continue
        result = inference_detector(model, frames[fid])
        if len(result[0][0]) == 0:
            continue
        mask = numpy.zeros(result[1][0][0].shape, dtype=bool)
        for bid in range(len(result[0][0])):
            if result[0][0][bid][4] < score_thr:
                continue
            mask = numpy.add(mask, result[1][0][bid])
        img = frames[fid].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(resolution[1]):
            for j in range(resolution[0]):
                if not mask[i][j]:
                    img[i][j] = [gray[i][j]]*3
        vwriter.write(img)
    vwriter.release()

if __name__ == '__main__':
    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    mod_file = sys.argv[3]
    conf_file = sys.argv[4]
    threshold = float(sys.argv[5])
    device="cuda:0"
    if len(sys.argv) > 6:
        device = sys.argv[6]
    mask_draw(src_file, dst_file, mod_file, conf_file, threshold, device)
