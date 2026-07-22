# @author Maria Lesniewski
# @file generator.py
# @desc this is the "real" generator. We're writing an algorithm and seeing if diffusion can do as well

import numpy as np
import cv2 # This is opencv 


# generate_circle_doodle(imglength)
def generate_circle_doodle(size=128):
    """
    Generate the circle doodle image of size size x size
    Places one big circle randomly on the page,
    then a random number of small ones on the exterior evenly spaced
    small ones have random sizes 

    Args:
        size (int): Number of pixels in each axis

    Returns:
        np.ndarray : A (imglength, imglength) uint8 array representing the image
        values range from 0 (black) to 255 (white).

    """
    img = np.zeros((size, size), dtype=np.uint8)

    # --- Big circle ---
    cx, cy = np.random.randint(0, size, 2) # center
    R = np.random.randint(size//8, size//2) # size
    cv2.circle(img, (cx, cy), R, 255, 1)

    # --- Small circles along perimeter ---
    num_small = np.random.randint(8, 20)
    angles = np.linspace(0, 2*np.pi, num_small, endpoint=False)

    small_circles = []
    for a in angles:
        r = np.random.randint(5, 10)

        x = int(cx + (R + r + 2) * np.cos(a))
        y = int(cy + (R + r + 2) * np.sin(a))

        # check overlap
        ok = True
        for (px, py, pr) in small_circles:
            if np.hypot(px-x, py-y) < pr + r + 2:
                ok = False
                break

        if ok:
            small_circles.append((x, y, r))
            cv2.circle(img, (x, y), r, 255, 1)

    # --- Speckle fill ---
    mask = np.zeros_like(img)

    # big circle mask
    cv2.circle(mask, (cx, cy), R, 255, -1)

    # subtract small circles
    for (x, y, r) in small_circles:
        cv2.circle(mask, (x, y), r, 0, -1)

    # speckles
    ys, xs = np.where(mask == 255)
    idx = np.random.choice(len(xs), size=len(xs)//10)

    for i in idx:
        img[ys[i], xs[i]] = 255

    return img


def generate_circle_doodle2(size=128, num_big=2):
    img = np.zeros((size, size), dtype=np.uint8)

    big_circles = []

    # --- Step 1: multiple big nonintersecting circles ---
    # Determine size of big circle
    big_R = np.random.randint(1, size // 4) ## Semi-Uniform big circles
    small_R = big_R // 3
    small_R = small_R + 1*(small_R == 0)
    print(big_R)
    num_big = size // (big_R)
    print(num_big)
    accepted_centerx = []
    accepted_centery = []
    for _ in range(num_big):
        distances = []
        cx, cy = np.random.randint(size/10, size, 2)
        for i in range(len(accepted_centerx)):
            distances.append(np.sqrt((cx - accepted_centerx[i])**2 + (cy - accepted_centery[i])**2)) # Center Distances
        brejected = 0
        for i in range(len(accepted_centerx)):
            if (distances[i] <= (2*big_R + 2*small_R + 2)): # No overlap + Shell for small circles and not touch big 
                brejected = 1
                print("broke!")
                break

        if not brejected:
            accepted_centerx.append(cx)
            accepted_centery.append(cy)
            big_circles.append((cx, cy, big_R))
            cv2.circle(img, (cx, cy), big_R, 255, 1)
    #return img
    
    # --- Step 2: small circles around each big circle ---
    for (cx, cy, R) in big_circles:

        #num_small = np.random.randint(6, 18)
        num_small = 3*big_R*2 // (2 * small_R)
        angles = np.linspace(0, 2 * np.pi, num_small-1, endpoint=False)

        small_circles = []

        for a in angles:
            #r = np.random.randint(3, 8)

            # place on perimeter
            radius_offset = R + small_R + 2
            x = int(cx + radius_offset * np.cos(a))
            y = int(cy + radius_offset * np.sin(a))

            # overlap check (local only)
            ok = True
            for (px, py, pr) in small_circles:
                if np.hypot(px - x, py - y) < pr + small_R + 2:
                    ok = False
                    break

            if ok:
                small_circles.append((x, y, small_R))
                cv2.circle(img, (x, y), small_R, 255, 1)
        # --- Step 3: speckle fill per big circle ---
        mask = np.zeros_like(img)

        # big circle region
        cv2.circle(mask, (cx, cy), R, 255, -1)

        # subtract small circles
        for (x, y, r) in small_circles:
            cv2.circle(mask, (x, y), small_R, 0, -1)

        ys, xs = np.where(mask == 255)

        if len(xs) > 0:
            idx = np.random.choice(len(xs), size=len(xs) // 10, replace=False)
            for i in idx:
                img[ys[i], xs[i]] = 255

    return img