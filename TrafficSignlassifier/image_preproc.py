import bisect
from skimage import data, img_as_float, img_as_ubyte
from skimage import exposure
import numpy as np

def imadjust_1(src, dst, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image buonds
    # return : output img

    tol = max(0, min(100, tol))
    src = src.astype(np.ubyte)
    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst

def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    dst = np.zeros(src.shape, dtype=np.ubyte)
    #dst = src.copy()
    for i in range(src.shape[2]):
        imadjust_1(src[:,:,i], dst[:,:,i], tol, vin, vout)
    return dst  
    
def equalizeHistHSV(src):
    dst = src.copy()
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    for i in range(2, dst.shape[2]):
        dst[:,:,i] = cv2.equalizeHist(dst[:,:,i])
    dst = cv2.cvtColor(dst, cv2.COLOR_HSV2RGB)
    return dst
        
def adapthisteq(src):
    img = img_as_float(src)
    dst = exposure.equalize_adapthist(img, clip_limit=0.03)
    dst = img_as_ubyte(dst)
    return dst