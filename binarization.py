import sys
import cv2 as cv
import os

def get_high(mask):
  cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

  chigh = None
  cy = 10000
  for c in cnts:
    x,y,w,h  = cv.boundingRect(c)
    s = min(w, h)
    if s < 15:
      continue

    if y < cy:
      r = max(w, h) / s
      if r > 1.5:
        continue

      cy = y
      chigh = c
  return chigh

f = 'sample_data/-8FLF-osZmA.mp4'
vs = cv.VideoCapture(f)

backSub = cv.createBackgroundSubtractorMOG2()

n = 0
while(True):
    ret, frame = vs.read()

    if not ret or frame is None:
        break

    mask = backSub.apply(frame)
    mask = cv.GaussianBlur(mask, (7, 7),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    cmask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    chigh = get_high(mask)
    if chigh is not None:
        rx,ry,rw,rh  = cv.boundingRect(chigh)
        cut = mask[ry : ry + rh, rx : rx + rw]
        cv.imwrite("segmentation_out/grey/b-{}.jpg".format(n), cut)
        cut_f = frame[ry : ry + rh, rx : rx + rw]
        cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut)
        cv.imwrite("segmentation_out/color/-8FLF-osZmA-c-{}.jpg".format(n), cut_c)

    print(n)
    n += 1