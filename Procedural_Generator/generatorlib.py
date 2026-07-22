# @author Maria Lesniewski
# @file generator.py
# @desc this is the "real" generator. We're writing an algorithm and seeing if diffusion can do as well

import numpy as np
import cv2 # This is opencv 

class CircleDoodle:
    """
    Generates non-overlapping circular patterns on a discrete grid.

    The drawing algorithm (fill_space) uses two boolean masks:
    space_tracker: tracks occupied blocks from each circle unit
    negative_space: tracks regions outside of circles within each unit

    Attributes:
        size (int): Grid dimension.
        image (np.ndarray) : 
        regionfull (np.ndarray): Regional Occupancy map for avoiding intersections
        negativespace (np.ndarray):  Local Occupancy map for coloring negative space

    Example:
        doodle = CircleDoodle(size=300)
        doodle.fillspace()
    """
    def __init__(self, size=300):
        self.size = size # Canvas length and width
        self.image = np.zeros((size,size), dtype=np.uint8) # Doodle content on canvas
        self.regionfull = np.zeros((size,size), dtype=bool) # tracks occupied regions
        self.negativespace = np.zeros((size, size), dtype=bool) # Tracks space outside of cv2 circles

    def fill_space(self):
        """
        Fills the image with white circle units and speckled negative space. 
        Circle units are random in size, surrounded by smaller circles and do not intersect
        """

        # Start by filling in the units with big circles that are at most 1/5 of the page 
        # and at least 1/20th of the page
        quit = False
        trycap = 200
        ntry = 0
        while (not quit) and (ntry < trycap):
            self.generate_circle_doodle_unit()
            ntry += 1
        
        self.fill_negative_space()
            

    def generate_circle_doodle_unit(self):
        """
        Places one big circle randomly on the drawing stored in image
        then a random number of small ones on the exterior evenly spaced
        small ones have random sizes 
        Do not let units intersect

        Returns: 
            False when no more circle units were successfully added
        """
        # Attempt to place a circle unit somewhere random on the page within the available space
        xavail, yavail = np.where(~self.regionfull)
        position_found = False
        trycap = 10
        ntry = 0
        while ((not position_found) and (ntry < trycap)):
            idx = np.random.randint(0, len(xavail))
            # Big Circle Center and Size
            cx = xavail[idx]
            cy = yavail[idx]
            R = np.random.randint(self.size//20, self.size//5)
            # Small circles on perimeter at random positons
            r = np.random.randint(R//30 + 1 , R//6)
            num_small = 2*np.pi * R // (2*r + 4) # offset so that small circles have some spacebetween

            position_found = not self.intersect(cx, cy, R, r)
            ntry += 1
        
        if (not position_found):
            # If we got out the while loop and have no position then we arent putting a big circle in 
            return False 
        
        self.add_doodle_unit_hitbox(cx, cy, R, r)
        self.add_local_hitbox(cx, cy, R)

        # Actually draw the circles into the calculated region
        # Big Circle
        cv2.circle(self.image, (cx, cy), R, 255, 1)
        # Small evenly spaced circles with no overlap
        #print("Hello!" + str(r))

        angles = np.linspace(0, 2*np.pi, int(num_small), endpoint=False)
        small_circles = []
        for a in angles:
            sx = int(cx + (R + r + 2) * np.cos(a)) # small circle centers
            sy = int(cy + (R + r + 2) * np.sin(a))

            # check overlap
            ok = True
            for (scx, scy, scr) in small_circles:
                # 
                if np.hypot(scx-sx, scy-sy) < scr + r + 2: 
                    ok = False
                    break

            if ok:
                small_circles.append((sx, sy, r))
                self.add_local_hitbox(sx, sy, r)
                cv2.circle(self.image, (sx, sy), r, 255, 1)
        

    
    def intersect(self, cx, cy, R, r):
        """
        Checks if the proposed circle_doodle_unit fits in the remaining circle doodle region

        Attributes: 
            cx - centerx coordinate of the region 
            cy - centery coordinate of the region
            R - Big Circle radius
            r - small circles radius

        Returns:
            boolean (True = intersection found, false = no intersection found) 
        """
        # Determine square region that circle doodle unit occupies
        xlo, xhi = self.calc_square_doodle_bound(cx, R, r)
        ylo, yhi = self.calc_square_doodle_bound(cy, R, r)

        # Trim to just the circle
        hitbox_radius = R + r + 1
        x, y = np.ogrid[xlo:xhi, ylo:yhi]
        circle_mask = (x - cx)**2 + (y - cy)**2 <= hitbox_radius**2

        region = self.regionfull[xlo:xhi, ylo:yhi][circle_mask]
        return np.any(region)
    
    def calc_square_doodle_bound(self, center, R, r):
        """
        Determines the boundaries of the circle doodle unit from center and circle radii

        Attributes: 
            center (int) - circle doodle unit center 
            R (int) - Big circle radius
            r (int) - little circles radius
        
        Returns:
            (int), (int) - the boundaries of the drawing that the doodle unit occupies
        """
        lo = np.maximum(center - 2*R - 2*r - 2, 0) # 2s shift to allow space between small and large circles
        hi = np.minimum(center + 2*R + 2*r + 2, self.size - 1)
        
        return lo, hi

    def add_doodle_unit_hitbox(self, cx, cy, R, r):
        """
        Marks the region of the unit doodle as full in the regionfull attribute to avoid future collisions
        """
        xlo, xhi = self.calc_square_doodle_bound(cx, R, r)
        ylo, yhi = self.calc_square_doodle_bound(cy, R, r)

        hitbox_radius = R + 2*r + 2 
        x, y = np.ogrid[xlo:xhi, ylo:yhi]
        circlemask = (x - cx)**2 + (y - cy)**2 <= hitbox_radius**2
        self.regionfull[xlo:xhi, ylo:yhi][circlemask] = True

    def add_local_hitbox(self, cx, cy, radius):
        """
        Marks locally what space is occupied by circles such that we know where the negative space of the image is
        """
        xlo, xhi = self.calc_square_doodle_bound(cx, radius, 0)
        ylo, yhi = self.calc_square_doodle_bound(cy, radius, 0)
        x, y = np.ogrid[xlo:xhi, ylo:yhi]

        circlemask = (x - cx)**2 + (y - cy)**2 <= radius**2
        self.negativespace[xlo:xhi, ylo:yhi][circlemask] = True

    def fill_negative_space(self):
        """
        Speckles 1/10 of the negative space 
        """
        xavail, yavail = np.where(~self.negativespace)
        
        sample_size = len(xavail) // 20
        indices = np.random.choice(len(xavail), sample_size, replace=False)
        xpoints = xavail[indices]
        ypoints = yavail[indices]

        self.image[ypoints, xpoints] = 255

        
