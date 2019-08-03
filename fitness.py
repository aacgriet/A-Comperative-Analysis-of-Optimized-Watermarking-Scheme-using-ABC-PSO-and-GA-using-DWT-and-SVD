

def func(k):
    import cv2
    import numpy as np
    import pywt
    from imgdb import im2double
    from sharp import sharpen
    from skimage.transform import rotate
    from skimage.util import random_noise
    from imblur import motion_blur
    from scipy import signal
    from skimage.measure import compare_ssim as ssim
    host = cv2.imread('lena_gray_512.tif')
    k = np.array(k)
    k = k / k.max(axis=0)

    host = cv2.cvtColor(host, cv2.COLOR_BGR2GRAY)
    host = cv2.resize(host, (512, 512))
    # cv2.imshow('Host Image',host)
    cv2.waitKey(2000)

    a = host
    coeffs = pywt.dwt2(a, 'haar')
    ll, (lh, hl, hh) = coeffs
    llu, lls, llvh = np.linalg.svd(ll, full_matrices=False)
    w = cv2.imread('default.jpeg')
    w = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
    w = cv2.resize(w, (64, 64))
    cv2.imshow('Watermark', w)
    cv2.waitKey(2000)
    wu, ws, wvh = np.linalg.svd(w, full_matrices=False)
    llsm = lls
    llsm[0:64] = lls[0:64] + ws * 0.001 * k
    llsm = np.array(llsm)
    llsm = np.diag(llsm)
    llnew = np.matmul(llu, np.matmul(llsm, llvh))
    coeffs2 = llnew, (lh, hl, hh)
    aw = pywt.idwt2(coeffs2, 'haar')
    data1 = ssim(a, aw)

    out = im2double(aw)
    aw = out
    cv2.imshow('OUTPUT', aw)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    def switch(num):
        if num == 1:
            aw1 = sharpen(aw)
            # cv2.imshow('sharpen',aw1)
            return aw1
        elif num == 2:
            aw1 = cv2.resize(aw, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
            # cv2.imshow('scaled',aw1)
            return aw1
        elif num == 3:
            aw1 = rotate(aw, 45)
            # cv2.imshow('rotated',aw1)
            return aw1
        elif num == 4:
            aw1 = random_noise(aw, mode='s&p', seed=None, clip=True)
            # cv2.imshow('salt and pepper noise',aw1)
            return aw1
        elif num == 5:
            aw1 = random_noise(aw, mode='gaussian', seed=None, clip=True)
            # cv2.imshow('Gaussian',aw1)
            return aw1
        elif num == 6:
            aw1 = motion_blur(aw)
            # cv2.imshow('Motion Blur',aw1)
            return aw1
        elif num == 7:
            aw1 = cv2.blur(aw, (10, 10))
            # cv2.imshow('Blurred Image',aw1)
            return aw1
        elif num == 8:
            aw1 = np.flipud(aw)
            # cv2.imshow('Verticle Flip',aw1)
            return aw1
        elif num == 9:
            aw1 = cv2.flip(aw, 0)
            # cv2.imshow('Horizontal Fllip',aw1)
            return aw1
        elif num == 10:
            aw1 = cv2.blur(aw, (5, 5))
            # cv2.imshow('Averaging',aw1)
            return aw1
        else:
            print('invalid option')
            quit()

    for num in range(1, 11):
        aw1 = switch(num)
        aw1 = aw1 * 511
        coeffs3 = pywt.dwt2(aw1, 'haar')
        llw, (lhw, hlw, hhw) = coeffs3
        aw1u, aw1s, aw1vh = np.linalg.svd(llw, full_matrices=False)
        s2 = aw1s[0:64] - lls[0:64]
        s2 = np.divide(s2, k)
        s2 = np.diag(s2)
        w1 = np.matmul(wu, np.matmul(s2, wvh))
        w1 = np.array(w1)
        w1 = cv2.resize(w1, (64, 64))
        out = im2double(w1)
        w1 = out
        # cv2.imshow('Extracted watermark image',w1)
        cv2.waitKey(2000)
        w = im2double(w)

        x = ssim(w, w1)
        print(x)
        z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        z[num - 1] = x
        num = num + 1
    for num in range(1, 11):
        sum = 0
        sum = sum + z[num - 1]
    f = 1 / ((sum) / 10 - data1)
    return f
