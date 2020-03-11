# SCROLL
Texture scrolling, how to scroll texture


PROJECT:
```
This library contains methods for scrolling pygame.Surface vertically and horizontally.
For each functionality ROLL/SCROLL, STACKING, TRANSPOSE) two solutions have been implemented
in order to access different data structure (buffer and array type).
```
METHODS:
```
1) Use of C-buffer data structure that provides the fastest algorithms and also the easiest
to understand (single loop). e.g. buffer[index] pointing to a specific pixel RGB or RGBA model
Note that we are using prange to increase algorithm speed.

2) Numpy arrays or memoryviewslice (memory buffer types) are also providing good performances and
gives easy access to the texture's pixels (with row and column indexing).
eg buffer[row, column, 0] pointing to a pixel RGB or RGBA model

Other methods such as TRANSPOSE and STACKING are not essential but could be very useful 
in certain circumstances when doing image processing.

All the algorithms have been coded for the model RGB or RGBA and for 24-bit and 32-bit
pygame texture. 8-bit format image will failed to load and an error message will be thrown
to your screen.
Other pixel model such as BGR and BGRA have not been tested but this, should not
be a great deal to adjust.
```
REQUIRMENT:
```
- python > 3.0
- numpy
- pygame with SDL version 1.2 (SDL version 2 untested)
  Cython
- A compiler such visual studio, MSVC, cgywin setup correctly
  on your system
```
BUILDING PROJECT:
```
Use the following command:
C:\>python setup_scroll.py build_ext --inplace
```

PYGAME SCROLL METHOD VS CYTHON:
```
scroll(dx=0, dy=0) -> None
Move the image by dx pixels right and dy pixels down.
dx and dy may be negative for left and up scrolls respectively.
Areas of the surface that are not overwritten retain their original pixel values.
Scrolling is contained by the Surface clip area.
It is safe to have dx and dy values that exceed the surface size.

pygame.Surface.scroll will push all the pixels outside the boundaries.
The Cython method push pixels outside the boundaries and create a loop effect by inserting
pixels on the opposite edges. As the result the animation will be smooth and continuous.

# TODO some functions can be twicked by passing empty array/buffer same size than original
buffer/array (when using the method in the loop). This will increase performance as
the function will not create a numpy.empty array each call.
```
