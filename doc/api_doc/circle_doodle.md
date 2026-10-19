# CircleDoodle API

`CircleDoodle` generates procedurally generated, black-and-white images containing non-overlapping circle patterns. Each image is represented by a NumPy array and can be annotated with properties such as circle count, circle size distribution, and image dimensions.

The class also supports saving generated images as `.npy` files, recording their annotations in a CSV database, and loading existing images and annotations back into a `CircleDoodle` object.

## Overview

The primary workflow is:

1. Initialize a `CircleDoodle` object.
2. Generate a circle pattern.
3. Populate the associated labels.
4. Save the image and its annotations.
5. Optionally, reload the image and compare objects.

## Initialization

```
CircleDoodle( 
    size=300, 
    fillpercent=1/20, 
    maxcircle=200, 
    filename="default.npy", 
    annotations="annotations.csv" 
    )
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `size` | `int` | `300` | Width and height of the square image in pixels. |
| `fill_percent` | `float` | `1/20` | Fraction of available negative space to fill with randomly sampled white pixels. (After circles are drawn) |
| `max_circle` | `int` | `200` | Maximum number of circle-unit placement attempts during generation. |
| `filename` | `str` | `"default.npy"` | Filename used when saving the generated image. |
| `annotations` | `str` | `"annotations.csv"` | CSV file used to store image annotations.|

**Returns:** An initialized `CircleDoodle` object.

Initialization creates a blank image and initializes the attributes used to track circle counts, occupied regions, and image statistics.

Note that `max_circle` controls placement attempts, not the final number of circles.

## Attributes

### Image and File Attributes

| Attribute | Type | Description |
|---|---|---|
| `size` | `int` | Image width and height. |
| `image` | `np.ndarray` | Grayscale image stored as a `size × size` NumPy array with `uint8` values. |
| `image_filename` | `str` | Image filename used for saving and annotation. |
| `access_filename` | `str` | Path used to access the saved image. |
| `annotations_file` | `str` | CSV file containing image annotations. |
| `category` | `str` | Image category, initialized to `"Procedural"`. |

### Circle Statistics

| Attribute | Type | Description |
|---|---|---|
| `ncircles` | `int` | Total number of visible circles, populated by `label_datapoint_with_ncircles()`. |
| `nbig` | `int` | Number of visible large circles. |
| `nsmall` | `int` | Number of visible small circles. |
| `area_of_circles` | `int` | Accumulated count of pixels marked as belonging to circle interiors. |
| `area_of_negative_space` | `int` | Tracked area outside circle interiors. |
| `fill_fraction` | `float` | Fraction of available negative space selected for speckling. |
| `max_circle_units` | `int` | Maximum number of circle-unit placement attempts. |
| `circle_offset` | `int` | Spacing parameter used when placing circles within a unit. |

### Internal Spatial Masks

| Attribute | Type | Description |
|---|---|---|
| `regionfull` | `np.ndarray` | Boolean mask used to prevent intersections between circle units. |
| `negativespace` | `np.ndarray` | Boolean mask tracking pixels occupied by circle interiors. |

These masks are used by the procedural generation algorithm to track occupied regions and determine where negative-space speckling can occur.

## Methods
### Methods for Image Generation (Synthesizing Features)
- `fill_space()`
  <details>
  <summary>Driver: Generates Image with multiple circles and speckles</summary>

    Usage:
    `fill_space()`

    Generates the image by attempting to place circle units and then filling a fraction of the remaining negative space with randomly sampled white pixels.

    Each circle unit consists of a large circle surrounded by smaller circles.

    **Returns:** `None`

    This method modifies the object's image, spatial masks, and circle statistics.
  </details>

- `generate_circle_doodle_unit()`

    <details>
    <summary>Attempts to place one large circle and perimeter circles within an image</summary>

    Usage:
    `generate_circle_doodle_unit()`

    Attempts to place one circle unit at a randomly selected location.

    The method checks for intersections with previously placed units before drawing the large circle and its surrounding smaller circles.

    **Returns:**
    - `False` if no valid placement is found within the allowed attempts.
    - `None` if a unit is successfully placed.
    </details>

- `intersect(cx, cy, R, r)`

    <details>
    <summary> 
    Checks whether a proposed circle unit intersects an occupied region. </summary>
    intersect(cx, cy, R, r)



    | Parameter | Description |
    |---|---|
    | `cx` | Proposed center x-coordinate. |
    | `cy` | Proposed center y-coordinate. |
    | `R` | Radius of the large circle. |
    | `r` | Radius of the surrounding small circles. |

    **Returns:** `bool`

    - `True` if an intersection is detected.
    - `False` if no intersection is detected.
    </details>

- `calc_square_doodle_bound(center, R, r)`

    <details>
    <summary> 
    Calculates a clipped coordinate interval around a proposed circle unit. </summary>

    Usage:
    ```
    calcsquaredoodle_bound(center, R, r)
    ```

    Calculates a clipped coordinate interval around a proposed circle unit.

    | Parameter | Description |
    |---|---|
    | `center` | Center coordinate along one axis. |
    | `R` | Large-circle radius. |
    | `r` | Small-circle radius. |

    **Returns:** A tuple `(lo, hi)` containing the calculated lower and upper bounds.

    This method is primarily used internally by the placement and masking routines.
    </details>

- `add_doodle_unit_hitbox(cx, cy, R, r)`

    <details>
    <summary>Updates the `regionfull` parameter with area occupied by the total circle unit (large circle + perimeter small circles) </summary>
    Usage:
    `adddoodleunit_hitbox(cx, cy, R, r)`


    Updates `regionfull` to mark the region reserved for a circle unit.

    This mask prevents subsequent circle units from being placed too close to existing units.

    **Returns:** `None`
    </details>

- `add_local_hitbox(cx, cy, radius)`

    <details>
    <summary> Updates `negativespace` to indicate where a single circle locally occupies </summary>

    Usage:
    `addlocalhitbox(cx, cy, radius)`

    Updates `negativespace` to mark the interior of a circle.

    The method also updates the tracked circle and negative-space area statistics.

    **Returns:** `None`
    </details>

- `fill_negative_space()`

    <details>
    <summary> Sets a fraction of the non-circle occupied image to white at random pixel locations</summary>

    Usage:
    `fillnegativespace()`

    Randomly selects a fraction of the pixels outside circle interiors and sets their image values to white (`255`).

    The fraction is controlled by `fill_fraction`.

    **Returns:** `None`
    </details>

### Methods for Labeling Images
- `label_datapoint_with_ncircles()`

    <details>
    <summary>adds a circle count to the object (post drawing)</summary>
    Usage:
    `labeldatapointwith_ncircles()`

    Updates the total circle count using the accumulated counts of large and small circles.

    **Returns:** A list containing:

    [ncircles, nsmall, nbig]


    This provides the primary circle-count label for supervised learning.
    </details>

- `label_datapoints_with_area()`

    <details>
    <summary>adds area of circles in drawing to the object (post drawing)</summary>
    Usage:
    `labeldatapointswith_area()`

    Calculates an area-based statistic from the image's negative-space mask.

    **Returns:** `float`

    The returned value is intended to represent the fraction of the image occupied by circles. See the implementation for the exact calculation.
    </details>


- `circle_is_visible(sx, sy, r)`

    <details>
    <summary>Checks whether a circle with radius r and center sx,sy is visible within the image map</summary>
    circleisvisible(sx, sy, r)


    Checks whether a circle with center `(sx, sy)` and radius `r` would have at least one pixel inside the image boundaries.

    **Returns:** `bool`

    This method is used during generation to avoid counting circles that are entirely outside the image.
    </details>

### Visualization, Equivalence Checking, Adding to Database

- `show_obj()`

    <details>
    <summary>print the drawing with Matplotlib</summary>
    show_obj()


    Prints the object's current statistics and displays the generated image using Matplotlib.

    **Returns:** `None`

    Useful for inspecting generated images and their associated labels during development.
    </details>

- `add_to_data_base(rootdir)`

    <details>
    addtodata_base(rootdir)


    <summary>Saves the image and appends its annotations to a CSV file.</summary>

    | Parameter | Type | Description |
    |---|---|---|
    | `rootdir` | `str` or `None` | Directory in which to save the image and annotation file. If `None`, the configured filenames are used directly. |

    The image is saved as a NumPy `.npy` file. The annotation CSV contains the following columns:

    | Column | Description |
    |---|---|
    | `access_filename` | Path used to access the saved image. |
    | `image_filename` | Image filename. |
    | `image_size` | Image width and height. |
    | `ncircles` | Total number of circles. |
    | `nbig` | Number of large circles. |
    | `nsmall` | Number of small circles. |
    | `area_of_circles` | Tracked circle-interior area. |
    | `area_of_negative` | Tracked negative-space area. |
    | `max_circle_units` | Maximum number of placement attempts. |
    | `fill_fraction` | Negative-space speckling fraction. |

    **Returns:** `None`

    The CSV header is written when the annotation file does not already exist.
    </details>

- `load_circle_obj_from_data(annotated_file, npy_access_filename)`

    <details>
    <summary>Loads an image from a .csv file into current object</summary>
    loadcircleobjfromdata(annotatedfile, npyaccess_filename)


    Loads an existing image and its annotations into the current object.

    | Parameter | Type | Description |
    |---|---|---|
    | `annotated_file` | `str` | Path to the CSV annotation file. |
    | `npy_access_filename` | `str` | Path to the image's `.npy` file. |

    The method finds the matching annotation row, loads the image, and restores the associated statistics.

    **Returns:** `None`

    The implementation expects exactly one matching annotation row. If no match or multiple matches are found, it prints an error and calls `exit()`.
    </details>

- `is_equivalent(other)`

    <details>
    <summary>Compares object to other, returns false is not the same</summary>
    is_equivalent(other)


    Compares two `CircleDoodle` objects using selected scalar attributes and the image array.

    | Parameter | Type | Description |
    |---|---|---|
    | `other` | `CircleDoodle` | Object to compare against. |

    **Returns:** `bool`

    Returns `True` if the compared attributes and image arrays are equivalent; otherwise, returns `False`.

    The comparison currently excludes internal spatial masks, `category`, and `circle_offset`.
    </details>


# Example Class Usage

## Generate and Save an Image

```
from generator.circle_doodle import CircleDoodle


#Initialize an image generator
doodle = CircleDoodle( size=300, fillpercent=0.05, maxcircle=200, filename="example.npy", annotations="annotations.csv" )

#Generate the image
doodle.fill_space()

#Populate the circle-count labels
doodle.labeldatapointwith_ncircles()

#Save the image and annotations
doodle.addtodata_base("data")

#Inspect the result
doodle.show_obj()
```

## Load an Image and Check Equivalence 
```
# Load by calling the .csv database and the specific feature tensor
doodle_obj2.load_circle_obj_from_data("data_folder/annotations.csv","data_folder/name.npy")

#Check if another object is the same as your root object
if (doodle_obj.is_equivalent(doodle_obj2)):
    print("The two objects are equivalent")
else:
    print("NOPE. They're different")
```


# See Also
- [CircleData](circle_data.md) - CircleForge's class for storing and indexing generated CircleDoodles, used for model training and analysis. 
