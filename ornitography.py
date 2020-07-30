#!/bin/env python3
import cv2
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to the input video file", type=str)
    parser.add_argument("-o", "--output", help="path to the output image", type=str)
    parser.add_argument("-i", "--interval", help="delay between frames (s)", type=float, default=0.2)
    parser.add_argument("--threshold", help="pixel value threshold for composition", type=int, default=10)
    parser.add_argument("--start", help="trim the beginning of the video (s)", type=float, default=0.)
    parser.add_argument("--stop", help="trim the end of the video (s)", type=float)
    return parser.parse_args()

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if (cap.isOpened()== False):
      print("Error opening video stream or file")
      exit(1)

    read_success, background = cap.read()

    if args.output is not None:
        image_out = args.output
    else:
        pre, ext = os.path.splitext(args.video)
        image_out = pre + '.png'
    out_dir = os.path.dirname(image_out)

    if not read_success:
        print("Error reading video stream or file")
        exit(1)

    # Read until video is completed
    started_compose = False
    image_idx = 1
    read_success, frame = cap.read()
    blur = lambda im: cv2.GaussianBlur(cv2.GaussianBlur(im, (3,3), 0), (5,5), 0)
    kernel = np.ones((5,5), np.uint8)
    erode = lambda im: cv2.erode(im, kernel, iterations = 1)
    dilate = lambda im: cv2.dilate(im, kernel, iterations = 1)
    while(cap.isOpened() and read_success):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        if not started_compose and current_sec > args.start:
            # set the background and start composition
            started_compose = True
            background = frame.copy()
            background_gray = blur(frame_gray.copy())
            composed_frame = background.copy()
        elif started_compose and current_sec > args.start + args.interval * image_idx:
            # add current frame to composition
            diff_background = dilate(erode(cv2.absdiff(background_gray, blur(frame_gray))))
            # mask = cv2.adaptiveThreshold(diff_background, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            mask = diff_background > args.threshold

            composed_frame[mask] = frame[mask]

            cv2.imwrite(os.path.join(out_dir, "composed_{}.png".format(image_idx)), composed_frame)
            cv2.imwrite(os.path.join(out_dir, "gray_diff_{}.png".format(image_idx)), diff_background)
            cv2.imwrite(os.path.join(out_dir, "mask_{}.png".format(image_idx)), mask.astype(np.uint8) * 255)
            image_idx += 1
        elif args.stop > 0 and current_sec > args.stop:
            break
        read_success, frame = cap.read()

    cap.release()

    cv2.imwrite(image_out, composed_frame)
    print("Saved composed image to {}".format(image_out))

main()
