# @author Maria Lesniewski
# @file circle_doodle.py
# @desc This class houses the info about each image in the database, and fills it in to a database

import numpy as np
import cv2 # This is opencv 
import matplotlib.pyplot as plt
import os 
import csv
import pandas as pd
from pathlib import Path
class CircleDoodle:
    """
    Generates non-overlapping circular patterns on a discrete grid.

    The drawing algorithm (fill_space) uses two boolean masks:
    space_tracker: tracks occupied blocks from each circle unit
    negative_space: tracks regions outside of circles within each unit

    Attributes:
        size (int): Grid dimension.
        image (np.ndarray) : black and white pixel demarkations
        image_filename (str) : file for reading/writing .npy
        regionfull (np.ndarray): Regional Occupancy map for avoiding global unit intersections
        negativespace (np.ndarray):  Local Occupancy map for coloring negative space
        ncircles (int) : total number of circles
        nbig (int) : total number of big circles placed
        nsmall (int) : total number of small circles encompassing large circles
        area_of_circles (int) : pixel space enclosed by circle
        area_of_negative_space (int) : pixel space excluded from circles
        fill_fraction (float) : fraction of excluded space to be speckled with black points
        category (str) : categorical label for image, here: procedurally generated. (As opposed to later when we build a generative model)
        max_circle_units (int) : number of tries the procedural algorithm will make to place a circle unit
        circle_offset (int) : Space between small and large circles in each circle unit, also space between small circles surrounding large circle 
        access_filename (str) : root directory + filenamepath 


    Example:
        doodle = CircleDoodle(size=300) # Creates Doodle Structure
        doodle.fillspace() # Draws actual image / fills attributes
        doodle.add_to_data_base() # Adds image path (saves as .npy) and labels to a csv data file
    """
    def __init__(self, size : int = 300, fill_percent : float = 1/20, max_circle : int = 200, filename : str = "default.npy", annotations : str = "annotations.csv" ):
        self.size : int = size # Canvas length and width
        self.image : np.ndarray = np.zeros((size,size), dtype=np.uint8) # Doodle content on canvas
        self.image_filename : str = str(filename)
        self.access_filename : str = str(filename)
        self.regionfull : int = np.zeros((size,size), dtype=bool) # tracks occupied regions including offset space and between (circular holes)
        self.negativespace : int = np.zeros((size, size), dtype=bool) # Tracks space outside of cv2 circles excluding offsets / between space (circled circles)
        self.ncircles : int = 0 # Total circles = Big + small
        self.nbig : int = 0 # Total Big 
        self.nsmall : int = 0 # Total Small
        self.area_of_circles : int = 0
        self.area_of_negative_space : int = size*size
        self.fill_fraction : float = fill_percent
        self.category : str = "Procedural"
        self.max_circle_units : int = max_circle
        self.circle_offset : int = 2
        self.annotations_file = annotations

    def fill_space(self):
        """
        Fills the image with white circle units and speckled negative space. 
        Circle units are random in size, surrounded by smaller circles and do not intersect
        """

        # Start by filling in the units with big circles that are at most 1/5 of the page 
        # and at least 1/20th of the page
        ntry = 0
        while (ntry < self.max_circle_units):
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
            R = np.random.randint(self.size//20 + 1, self.size//3 + 2)
            # Small circles on perimeter at random positons
            r = np.random.randint(R//30 + 1 , R//6 + 2)
            num_small = 2*np.pi * (R + 2) // (2*r + 4) # R offset by sit space, r offset by double between small circle space

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
        if self.circle_is_visible(cx, cy, R):
            self.nbig += 1
        # Small evenly spaced circles with no overlap
        #print("Hello!" + str(r))

        angles = np.linspace(0, 2*np.pi, int(num_small), endpoint=False)
        small_circles = []
        for a in angles:
            sx = int(cx + (R + r + self.circle_offset) * np.cos(a)) # small circle centers
            sy = int(cy + (R + r + self.circle_offset) * np.sin(a))

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
                if self.circle_is_visible(sx, sy,r):
                    self.nsmall += 1
        

    
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
        hitbox_radius = R + r + self.circle_offset
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

        hitbox_radius = R + 2*r + self.circle_offset 
        x, y = np.ogrid[xlo:xhi, ylo:yhi]
        circlemask = (x - cx)**2 + (y - cy)**2 <= hitbox_radius**2
        self.regionfull[xlo:xhi, ylo:yhi][circlemask] = True
        #print(f"analytic is {hitbox_radius**2*np.pi}")
        # print(sum(self.regionfull[xlo:xhi, ylo:yhi][circlemask]))

    def add_local_hitbox(self, cx, cy, radius):
        """
        Marks locally what space is occupied by circles such that we know where the negative space of the image is
        """
        xlo, xhi = self.calc_square_doodle_bound(cx, radius, 0)
        ylo, yhi = self.calc_square_doodle_bound(cy, radius, 0)
        x, y = np.ogrid[xlo:xhi, ylo:yhi]

        circlemask = (x - cx)**2 + (y - cy)**2 <= radius**2
        self.negativespace[xlo:xhi, ylo:yhi][circlemask] = True
        self.area_of_circles += sum(self.negativespace[xlo:xhi, ylo:yhi][circlemask])
        self.area_of_negative_space -= sum(self.negativespace[xlo:xhi, ylo:yhi][circlemask])

    def fill_negative_space(self):
        """
        Speckles the fill fraction of the leftover negative space 
        """
        xavail, yavail = np.where(~self.negativespace)
        
        sample_size = (int) (len(xavail) * self.fill_fraction ) // 1
        indices = np.random.choice(len(xavail), sample_size, replace=False)
        xpoints = xavail[indices]
        ypoints = yavail[indices]

        self.image[ypoints, xpoints] = 255

    def label_datapoint_with_ncircles(self):
        self.ncircles = self.nsmall + self.nbig
        
        return [self.ncircles, self.nsmall, self.nbig]
    
    def label_datapoints_with_area(self):
        return (self.size**2 - np.sum(~self.negativespace))/self.size**2

    def show_obj(self):
        """
        Shows the image generated by the procedural algorithm along with the attributes current values
        """
        print(f"{self.size} x  {self.size} Doodle Object")
        print(f"Filename {self.access_filename}")
        print(f"N_circles=  {self.ncircles} = small: {self.nsmall} + big {self.nbig}")
        print(f"Negative space sum = {self.negativespace.sum()} = {self.area_of_negative_space}")
        print(f"Area of circles = {self.area_of_circles}")
        print(f"Negative + Circles = {self.area_of_circles + self.area_of_negative_space}")
        
        fig, ax = plt.subplots()

        ax.imshow(self.image, cmap='gray')
        ax.axis('off')
        plt.show(block=False)

    def add_to_data_base(self, rootdir):
        """
        Makes sure all labels associated with circle doodle object are populated
        Saves image and labels to a CSV sheet folder / image directory in the roodir passed
        """
        if rootdir != None:
            self.access_filename = rootdir + "/" + self.image_filename
            csv_filename = os.path.join(rootdir, self.annotations_file)
        else:
            self.access_filename = self.image_filename
            csv_filename = self.annotations_file
        np.save(self.access_filename, self.image)
        self.label_datapoints_with_area()
        self.label_datapoint_with_ncircles()

        file_exists = os.path.isfile(csv_filename)


        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow([
                    "access_filename", 
                    "image_filename",
                    "image_size",
                    "ncircles",
                    "nbig",
                    "nsmall",
                    "area_of_circles",
                    "area_of_negative",
                    "max_circle_units",
                    "fill_fraction"
                    ])
            writer.writerow([
                self.access_filename,
                self.image_filename,
                self.size,
                self.ncircles,
                self.nbig,
                self.nsmall,
                self.area_of_circles,
                self.area_of_negative_space,
                self.max_circle_units,
                self.fill_fraction
                ])
    def load_circle_obj_from_data(self, annotated_file, npy_access_filename):
        """
        Reads annotations.csv, finds the filename passed and loads existent stats into the object
        npy_access_filename should be a relative path
        """
        df = pd.read_csv(annotated_file)
        mask = (df["access_filename"].apply(Path) == Path(npy_access_filename)) # Note this assumes unique names/files
        if sum(mask) == 0:
            print(f"ERROR: Unable to find {npy_access_filename} entry in {annotated_file}")
            exit()
        elif sum(mask) > 1:
            print(f"ERROR: Found multiple {npy_access_filename} entries in {annotated_file}")
            print(f"mask entries {np.nonzero(mask)[0]}")
            exit()

        #row_data = df[mask]
        row_data = df[mask].iloc[0]
        self.access_filename = npy_access_filename
        self.image_filename = row_data["image_filename"]
        self.size = row_data["image_size"]
        self.image = np.load(npy_access_filename)

        self.ncircles = row_data["ncircles"]
        self.nbig = row_data["nbig"]
        self.nsmall = row_data["nsmall"]
        self.area_of_circles = row_data["area_of_circles"]
        self.area_of_negative_space = row_data["area_of_negative"]
        self.max_circle_units = row_data["max_circle_units"]
        self.fill_fraction = row_data["fill_fraction"]

    def is_equivalent(self, other):
        """
        Compare two CircleDoodle objects.

        Returns:
            True if all relevant doodle data is equivalent.
            False otherwise.
        """

        if not isinstance(other, CircleDoodle):
            return False

        # Scalar attributes
        scalar_attributes = [
            "size",
            "ncircles",
            "nbig",
            "nsmall",
            "area_of_circles",
            "area_of_negative_space",
            "fill_fraction",
     #       "category",
            "max_circle_units",
    #       "circle_offset"
        ]

        for attr in scalar_attributes:
            if not np.isclose(getattr(self, attr), getattr(other, attr)):
                print(f"Failed equivalence on attribute {attr}")
                print(f"self: {getattr(self, attr)} other: {getattr(other, attr)}")
                return False

        # Array attributes
        array_attributes = [
            "image",
        #    "regionfull",
        #    "negativespace"
        ]

        for attr in array_attributes:
            if not np.array_equal(getattr(self, attr), getattr(other, attr)):
                print(f"Failed equivalence on attribute {attr}")
                print(f"self: {getattr(self, attr)} other: {getattr(other, attr)}")
                return False

        return True

    def circle_is_visible(self, sx, sy, r):
        """
        Checks if a circle added with center sx, sy, and radius r is visible within the image
        """

        # Create a small block around the circle
        block_size = 2 * r + 3
        block = np.zeros((block_size, block_size), dtype=np.uint8)

        # Draw circle centered in the block
        cv2.circle(
            block,
            (r + 1, r + 1),
            r,
            255,
            1
        )

        # Coordinates of pixels belonging to circle
        ys, xs = np.where(block > 0)

        # Translate back to image coordinates
        xs = xs + sx - r - 1
        ys = ys + sy - r - 1

        # Check if any pixel is inside image
        return np.any(
            (xs >= 0) & (xs < self.size) &
            (ys >= 0) & (ys < self.size)
        )
