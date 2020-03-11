
###cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

"""
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

PROJECT:
This library contains necessary tools for scrolling texture (pygame.Surface)
vertically and/or horizontally to the desirable speed.
For each functionality ROLL/SCROLL, STACKING, TRANSPOSE) two methods have been
implemented to provide a solution for each data structure available at the time
of the functions calls.
If you are more efficient with numpy.ndarray in you program (using pygame.surfarray.pixels3d,
array3d, pixel_alpha, array_alpha...), prefer the methods using numpy.ndarray as argument.
If you are fond of C-buffer structure (using pygame.method get_view()), prefer the methods
using buffers as argument). 

METHODS:
1) Use of C-buffer data structure that provides the fastest algorithms and also the easiest
to understand (single loop). e.g. buffer[index] pointing to a specific pixel RGB or RGBA model
Note that we are using prange to increase algorithm speed.

2) Numpy arrays or memoryviewslice (memory buffer types) are also providing good performances and
an easy access to the pixel as it is very simple to refer to a specific pixel choosing row and
column indexing. eg buffer[row, column, 0] pointing to a pixel RGB or RGBA model

Other methods such as TRANSPOSE and STACKING are not essential but could be very useful 
in certain circumstances when doing image processing.

All the algorithms have been coded for the model RGB or RGBA and for 24-bit and 32-bit
pygame texture. 8-bit format image will failed to load and an error message will be thrown
to your screen, other pixel model such as BGR and BGRA have not been tested but this should not
be a great deal to adjust.

REQUIRMENT:
- python > 3.0
- numpy
- pygame with SDL version 1.2 (SDL version 2 untested)
  Cython
- A compiler such visual studio, MSVC, cgywin setup correctly
  on your system

BUILDING PROJECT:
Use the following command:
C:\>python setup_build.py build_ext --inplace

PYGAME SCROLL METHOD VS CYTHON:
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
the function will not create a numpy.empty array each time.


"""

# TODO: TEST IF IT IS SAFE TO HAVE DX OR DY > WIDTH OR HEIGHT

# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, dstack, full, ones,\
    asarray, ascontiguousarray
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")
    raise SystemExit

# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    print("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")
    raise SystemExit

cimport numpy as np


# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d
    from pygame.image import frombuffer

except ImportError:
    print("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")
    raise SystemExit

cimport numpy as np
import random
from random import randint
import math

from libc.math cimport sin, sqrt, cos, atan2, pi, round, floor, fmax, fmin, pi, tan, exp, ceil, fmod
from libc.stdio cimport printf
from libc.stdlib cimport srand, rand, RAND_MAX, qsort, malloc, free, abs

DEF OPENMP = True

# num_threads – The num_threads argument indicates how many threads the team should consist of.
# If not given, OpenMP will decide how many threads to use.
# Typically this is the number of cores available on the machine. However,
# this may be controlled through the omp_set_num_threads() function,
# or through the OMP_NUM_THREADS environment variable.
if OPENMP == True:
    DEF THREAD_NUMBER = 10
else:
    DEF THREAD_NUMNER = 1


# static:
# If a chunksize is provided, iterations are distributed to all threads ahead of
# time in blocks of the given chunksize. If no chunksize is given, the iteration
# space is divided into chunks that are approximately equal in size,
# and at most one chunk is assigned to each thread in advance.
# This is most appropriate when the scheduling overhead matters and the problem can be
# cut down into equally sized chunks that are known to have approximately the same runtime.

# dynamic:
# The iterations are distributed to threads as they request them, with a default chunk size of 1.
# This is suitable when the runtime of each chunk differs and is not known in advance and
# therefore a larger number of smaller chunks is used in order to keep all threads busy.

# guided:
# As with dynamic scheduling, the iterations are distributed to threads as they request them,
# but with decreasing chunk size. The size of each chunk is proportional to the number of
# unassigned iterations divided by the number of participating threads,
# decreasing to 1 (or the chunksize if provided).
# This has an advantage over pure dynamic scheduling when it turns out that the last chunks
# take more time than expected or are otherwise being badly scheduled, so that
# most threads start running idle while the last chunks are being worked on by
# only a smaller number of threads.

# runtime:
# The schedule and chunk size are taken from the runtime scheduling variable,
# which can be set through the openmp.omp_set_schedule() function call,
# or the OMP_SCHEDULE environment variable. Note that this essentially
# disables any static compile time optimisations of the scheduling code itself
# and may therefore show a slightly worse performance than when the same scheduling
# policy is statically configured at compile time. The default schedule is implementation defined.
# For more information consult the OpenMP specification [1].
DEF SCHEDULE = 'static'

# 
# ------------- INTERFACE --------------------
#

#*********************************************
#**********  METHOD ROLL/SCROLL  *************
#*********************************************

# ROLL BUFFER COMPATIBLE 24-BIT TEXTURE
# Refer to pygame doc in order to extract
# a buffer from an image get_view()
def scroll_buffer24(buffer_, w, h, dx=0, dy=0):
    return scroll_buffer24_c(buffer_, w, h, dx, dy)

# ROLL BUFFER COMPATIBLE 32-BIT TEXTURE
# Refer to pygame doc to extract
# a buffer from an image get_view()
def scroll_buffer32(buffer_, w, h, dx=0, dy=0):
    return scroll_buffer32_c(buffer_, w, h, dx, dy)

# ROLL ARRAY 3D TYPE (W, H, 3) NUMPY.UINT8
# Refer to pygame doc surfarray methods
# to get a 3d array from an image (pixels3d, array3d)
def scroll_array24(array_, dy=0, dx=0):
    return scroll_array24_c(array_, dy, dx)

# ROLL ARRAY 3D TYPE (W, H, 4) NUMPY.UINT8
# Refer to pygame doc surfarray methods
# to get a 3d array from an image (pixels3d, array3d)
def scroll_array32(array_, dy=0, dx=0):
    return scroll_array32_c(array_, dy, dx)

# ROLL ARRAY (INPUT RGB + ALPHA)
# Return a 3d array type (w, h, 4)
# inputs : 3d array (w, h, 3) and alpha (w, h)
# Refer to pygame doc surfarray methods
# to get a 3d array from an image (pixels3d, array3d)
def scroll_array32m(array_, alpha_, dy=0, dx=0):
    return scroll_array32m_c(array_, alpha_, dy, dx)

# ---------- NUMPY
# USE NUMPY LIBRARY (NUMPY.ROLL METHOD)
# See pygame.Surface for more details
def roll_surface(surface_, dx=0, dy=0):
    return roll_surface_c(surface_, dx, dy)

def roll_array(array_, dx=0, dy=0):
    return roll_array_c(array_, dx, dy)
# ----------------

# ROLL ARRAY (lateral and vertical)
# Identical algorithm (scroll_array) but
# returns a tuple (surface, array)
# See pygame.Surface for more details
def scroll_surface24(surface, dy=0, dx=0):
    return scroll_surface24_c(surface, dy, dx)

# ROLL IMAGE 32-bit
# See pygame.Surface for more details
def scroll_surface32(surface, dy=0, dx=0):
    return scroll_surface32_c(surface, dy, dx)

#*********************************************
#**********  METHOD STACKING  *************
#*********************************************

# ---------- ARRAY

# STACK RGB & ALPHA ARRAY VALUES,
# RETURN A PYTHON OBJECT
# Twice faster than numpy.dstack
def stack_object(rgb_array_, alpha_, transpose):
    return stack_object_c(rgb_array_, alpha_, transpose)

# STACK RGB & ALPHA ARRAY VALUES 
# RETURN A MEMORYVIEW (slightly faster than stack_object)
# Twice faster than numpy.dstack
def stack_mem(rgb_array_, alpha_, transpose):
    return stack_mem_c(rgb_array_, alpha_, transpose)

# UN-STACK RGBA ARRAY VALUES
def unstack_object(rgba_array):
    return unstack_object_c(rgba_array)
# ------------------

# ----------- BUFFER

# STACK RGB AND ALPHA BUFFERS
def stack_buffer(rgb_array_, alpha_, w, h, transpose):
    return stack_buffer_c(rgb_array_, alpha_, w, h, transpose)
    
# UN-STACK/SPLIT RGBA BUFFER WITH RGBA INTO
# (RGB BUFFER & ALPHA_BUFFER)
def unstack_buffer(rgba_buffer_, w, h):
    return unstack_buffer_c(rgba_buffer_, w, h)
# -------------------

#*********************************************
#**********  METHOD TRANSPOSE  *************
#*********************************************

# TRANSPOSE ROWS AND COLUMNS ARRAY (W, H, 3)
# !! Method slower than numpy.transpose(1, 0, 2)
def transpose24(rgb_array_):
    return transpose24_c(rgb_array_)

# TRANSPOSE ROWS AND COLUMNS ARRAY (W, H, 4)
# !! Method slower than numpy.transpose(1, 0, 2)
def transpose32(rgb_array_):
    return transpose32_c(rgb_array_)

# TRANSPOSE/ FLIP BUFFER (compatible 24-bit)
# method faster than transpose24
# No known equivalence with numpy
def vfb24(source, target, width, height):
    return vfb24_c(source, target, width, height)

# TRANSPOSE / FLIP BUFFER (compatible 32-bit)
# method faster than transpose32
# No known equivalence with numpy
def vfb32(source, target, width, height):
    return vfb32_c(source, target, width, height)

# ----------------END INTERFACE -----------------
#
#
#
#
#
# ----------------IMPLEMENTATION ----------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char[:, :, ::1] transpose24_c(unsigned char[:, :, :] rgb_array_):
    """
    Transpose array type (w, h, 3) rows and columns (similar to numpy.transpose(1, 0, 2)
    
    :param rgb_array_: numpy.ndarray (w, h, 3) uint8 containing RGBA values 
    :return: Return a contiguous memoryslice (numpy.ndarray (w, h, 3) uint8)
    filled with RGB pixel values
    """
    cdef int w, h
    try:
        w, h = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char[:, :, ::1] new_array =  empty((h, w, 3), dtype=uint8)
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                new_array[j, i, 0] = rgb_array_[i, j, 0]
                new_array[j, i, 1] = rgb_array_[i, j, 1]
                new_array[j, i, 2] = rgb_array_[i, j, 2]
    return new_array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char[:, :, ::1] transpose32_c(unsigned char[:, :, :] rgb_array_):
    """
    Transpose array type (w, h, 3) rows and columns (similar to numpy.transpose(1, 0, 2)
    
    :param rgb_array_: numpy.ndarray (w, h, 4) uint8 containing RGBA values 
    :return: return a contiguous memoryslice (numpy.ndarray (w, h, 4) uint8)
    filled with RGBA pixel values
    """
    cdef int w, h
    try:
        w, h = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char[:, :, ::1] new_array =  empty((h, w, 4), dtype=uint8)
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                new_array[j, i, 0] = rgb_array_[i, j, 0]
                new_array[j, i, 1] = rgb_array_[i, j, 1]
                new_array[j, i, 2] = rgb_array_[i, j, 2]
                new_array[j, i, 3] = rgb_array_[i, j, 3]
    return new_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef stack_object_c(unsigned char[:, :, :] rgb_array_,
                    unsigned char[:, :] alpha_, bint transpose=False):
    """
    Stack RGB pixel values together with alpha values and return a python object,
    numpy.ndarray (faster than numpy.dstack)
    If transpose is True, transpose rows and columns of output array.
    
    :param transpose: boolean; Transpose rows and columns
    :param rgb_array_: numpy.ndarray (w, h, 3) uint8 containing RGB values 
    :param alpha_: numpy.ndarray (w, h) uint8 containing alpha values 
    :return: return a contiguous numpy.ndarray (w, h, 4) uint8, stack array of RGBA pixel values
    The values are copied into a new array.
    """
    cdef int width, height
    try:
        width, height = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char[:, :, ::1] new_array =  numpy.empty((width, height, 4), dtype=uint8)
        unsigned char[:, :, ::1] new_array_t =  numpy.empty((height, width, 4), dtype=uint8)
        int i=0, j=0
    # Equivalent to a numpy.dstack
    with nogil:
        # Transpose rows and columns
        if transpose:
            for j in prange(0, height, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for i in range(0, width):
                    new_array_t[j, i, 0] = rgb_array_[i, j, 0]
                    new_array_t[j, i, 1] = rgb_array_[i, j, 1]
                    new_array_t[j, i, 2] = rgb_array_[i, j, 2]
                    new_array_t[j, i, 3] =  alpha_[i, j]

        else:
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, height):
                    new_array[i, j, 0] = rgb_array_[i, j, 0]
                    new_array[i, j, 1] = rgb_array_[i, j, 1]
                    new_array[i, j, 2] = rgb_array_[i, j, 2]
                    new_array[i, j, 3] =  alpha_[i, j]

    return asarray(new_array) if transpose == False else asarray(new_array_t)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

# TODO try to declare rgb_array_ and alpha as memoryviewslice instead 

cdef unsigned char[::1] stack_buffer_c(rgb_array_, alpha_, int w, int h, bint transpose=False):
    """
    Stack RGB & ALPHA memoryviewslice C-buffers structures together.
    If transpose is True, the output memoryviewslice is flipped.
    
    :param h: integer; Texture height
    :param w: integer; Texture width
    :param transpose: boolean; Transpose rows and columns (default False)
    :param rgb_array_: Memoryviewslice or pygame.BufferProxy (C-buffer type) representing the texture
    RGB values filled with uint8
    :param alpha_:  Memoryviewslice or pygame.BufferProxy (C-buffer type) representing the texture
    alpha values filled with uint8 
    :return: Return a contiguous memoryviewslice representing RGBA pixel values
    """
   
    cdef:
        int b_length = w * h * 3
        int new_length = w * h * 4
        unsigned char [:] rgb_array = rgb_array_
        unsigned char [:] alpha = alpha_
        unsigned char [::1] new_buffer =  numpy.empty(new_length, dtype=numpy.uint8)
        unsigned char [::1] flipped_array = numpy.empty(new_length, dtype=numpy.uint8)
        int i=0, j=0, ii, jj, index, k 
        int w4 = w * 4
    
    with nogil:
       
        for i in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                ii = i // 3
                jj = ii * 4
                new_buffer[jj]   = rgb_array[i]
                new_buffer[jj+1] = rgb_array[i+1]
                new_buffer[jj+2] = rgb_array[i+2]
                new_buffer[jj+3] = alpha[ii]
                
        if transpose:
            for i in prange(0, w4, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, h):
                    index = i + (w4 * j)
                    k = (j * 4) + (i * h)
                    flipped_array[k    ] = new_buffer[index    ]
                    flipped_array[k + 1] = new_buffer[index + 1]
                    flipped_array[k + 2] = new_buffer[index + 2]
                    flipped_array[k + 3] = new_buffer[index + 3]
            return flipped_array
        
    return new_buffer



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char[:, :,::1] stack_mem_c(unsigned char[:, :, :] rgb_array_,
                                         unsigned char[:, :] alpha_, bint transpose=False):
    """
    Stack RGB values together with alpha values and return a memoryslice 
    *faster than (numpy.dstack & stack_object_c)
    if transpose is True, flip the output array
    
    :param transpose: boolean; Transpose rows and columns
    :param rgb_array_: numpy.ndarray (w, h, 3) uint8 containing RGB values 
    :param alpha_: numpy.ndarray (w, h) uint8 containing alpha values 
    :return: return a contiguous memoryviewslice representing a 3d
    numpy.ndarray (w, h, 4) filled with RGBA uint8
    """
    cdef int width, height
    try:
        width, height = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char[:, :, ::1] new_array =  numpy.empty((width, height, 4), dtype=uint8)
        unsigned char[:, :, ::1] new_array_t =  numpy.empty((height, width, 4), dtype=uint8)
        int i=0, j=0
    # Equivalent to a numpy.dstack
    with nogil:
        if transpose:
            for j in prange(0, height, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for i in range(0, width):
                    new_array_t[j, i, 0] = rgb_array_[i, j, 0]
                    new_array_t[j, i, 1] = rgb_array_[i, j, 1]
                    new_array_t[j, i, 2] = rgb_array_[i, j, 2]
                    new_array_t[j, i, 3] =  alpha_[i, j]

        else:
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, height):
                    new_array[i, j, 0] = rgb_array_[i, j, 0]
                    new_array[i, j, 1] = rgb_array_[i, j, 1]
                    new_array[i, j, 2] = rgb_array_[i, j, 2]
                    new_array[i, j, 3] =  alpha_[i, j]

    return asarray(new_array) if transpose == False else asarray(new_array_t)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unstack_object_c(unsigned char[:, :, :] rgba_array_):
    """
    Un-stack RGBA values and return RGB & ALPHA equivalent numpy.ndarray. 
    
    :param rgba_array_: numpy.ndarray (w, h, 4) uint8 containing RGBA values 
    :return: return a contiguous 3d numpy.ndarray (w, h, 3) filled with RGB uint8 and 2d
    contiguous numpy array (w, h) filled with uint8 alpha values.
    """
    cdef int w, h, d
    try:
        w, h, d = (<object> rgba_array_).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    assert d==4, 'Invalid depth for arguement rgba_array_, must be 4 got %s ' % d
    
    cdef:
        unsigned char[:, :, ::1] rgb_array = numpy.empty((w, h, 3), numpy.uint8)
        unsigned char[:, ::1] alpha_array =  numpy.empty((w, h), dtype=uint8)
        int i=0, j=0
        
    with nogil:
        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(0, h):
                rgb_array[i, j, 0] = rgba_array_[i, j, 0]
                rgb_array[i, j, 1] = rgba_array_[i, j, 1]
                rgb_array[i, j, 2] = rgba_array_[i, j, 2]
                alpha_array[i, j] =  rgba_array_[i, j, 3]
                
    return numpy.asarray(rgb_array), numpy.asarray(alpha_array)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unstack_buffer_c(unsigned char [:] rgba_buffer_, int w, int h):
    """
    Un-stack RGBA values and return RGB & ALPHA equivalent memoryviewslice. 

    Buffer equivalence:
    array1 = numpy.array(rgb_buffer, dtype=uint8)
    array2 = numpy.array(image32bit.get_view('3'), dtype=uint8).transpose(1, 0, 2)
    array2 = array2.flatten(order='C')
    
    alpha1 = numpy.frombuffer(alpha_buffer, uint8)
    alpha2 = numpy.array(image32bit.get_view('a'), uint8).transpose(1, 0)
    alpha2 = alpha2.flatten(order='C')
        
    :param rgba_buffer_: pygame.ProxyBuffer containing RGBA values
    :param w:integer; image width
    :param h:integer; image height
    :return: Return a python tuple object with RGB & ALPHA contiguous Memoryviewslice 
    """

    cdef int b_length
    try:
        b_length = len(rgba_buffer_)
    except (ValueError, pygame.error) as e:
        raise ValueError('\Buffer not understood.')

    if b_length != w * h * 4:
        raise ValueError("Buffer length does not match image dimensions.")
    
    cdef:
        int i, ii, jj
        unsigned char [::1] rgb_buffer = numpy.empty(w * h * 3, numpy.uint8)
        unsigned char [::1] alpha_buffer = numpy.empty(w * h, numpy.uint8)
        unsigned char [:] rgba_buffer = rgba_buffer_# numpy.frombuffer(rgba_buffer_, numpy.uint8)
              
    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            jj = (i >> 2)
            ii = jj * 3
            rgb_buffer[ii  ] = rgba_buffer[i]      # Red
            rgb_buffer[ii+1] = rgba_buffer[i+1]    # Green
            rgb_buffer[ii+2] = rgba_buffer[i+2]    # Blue
            alpha_buffer[jj] = rgba_buffer[i+3]    # Alpha
                
    return rgb_buffer, alpha_buffer


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_buffer24_c(unsigned char [:] buffer_, int w, int h, int dy, int dx):
    """
    Scroll buffer vertically and/or horizontally (compatible with texture 24 bit)
    
    :param buffer_: numpy.ndarray type (w, h, 3) filled with RGB pixel values (uint8)
    :param w: integer; Texture width
    :param h: integer; Texture height
    :param dy: integer; dy > 0 scroll down, dy < 0 scroll up
    :param dx: integer; dx > 0 scroll right, dx < 0 left scroll
    :return : return a contiguous memoryslice representing the texture
    in a C-buffer structure 
    """
    
    cdef:
        int i, ii, x, y
        int b_length, dx3 = dx * 3, dyw3= dy * w * 3
        int w3 = w * 3
        
    try:
        b_length = len(<object>buffer_)
    except:
        raise ValueError("Possibly wrong type for argument buffer_")

    assert b_length == w * h * 3, \
        "Buffer length is not compatible with a 24 bit buffer"
    cdef:
        unsigned char [::1] new_buffer = numpy.empty(b_length, numpy.uint8)

    if dx == 0 and dy==0:
        return buffer_

    with nogil:
        if dx != 0 and dy != 0:
            for i in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    ii = (i + dx3) % b_length
                    ii = (ii + dyw3) % b_length
                    if ii < 0:
                        ii = b_length + ii
                    new_buffer[ii]   = buffer_[i]
                    new_buffer[ii+1] = buffer_[i+1]
                    new_buffer[ii+2] = buffer_[i+2]
        else:
            if dx !=0:
                for i in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    ii = (i + dx3) % b_length
                    if ii < 0:
                        ii = ii + w3
                    new_buffer[ii]   = buffer_[i]
                    new_buffer[ii+1] = buffer_[i+1]
                    new_buffer[ii+2] = buffer_[i+2]
            elif dy !=0:
                for i in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    ii = (i + dyw3) % b_length
                    if ii < 0:
                        ii = ii + b_length
                    new_buffer[ii]   = buffer_[i]
                    new_buffer[ii+1] = buffer_[i+1]
                    new_buffer[ii+2] = buffer_[i+2]
                
    return new_buffer


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_buffer32_c(unsigned char [:] buffer_, int w, int h, int dy, int dx):
    """
    Scroll buffer vertically and/or horizontally (compatible with texture 32-bit)
    
    :param buffer_: numpy.ndarray type (w, h, 4) filled with RGBA pixel values (uint8)
    :param w: integer; Texture width
    :param h: integer; Texture height
    :param dy: integer; dy > 0 scroll down, dy < 0 scroll up
    :param dx: integer; dx > 0 scroll right, dx < 0 left scroll
    :return : return a contiguous memoryslice representing the texture
    in a C-buffer structure 
    """
    
    cdef:
        int i, ii, x, y
        int b_length, dx4 = dx * 4, dyw4 = dy * w * 4
        int w4 = w * 4
        
    try:
        b_length = len(<object>buffer_)
    except:
        raise ValueError("Possibly wrong type for argument buffer_")

    assert b_length == w * h * 4, \
        "Buffer length is not compatible with a 32-bit buffer,"\
        "expecting %s got %s " % (w * h * 4, b_length)
    
    cdef:
        unsigned char [::1] new_buffer = numpy.empty(b_length, numpy.uint8)

    if dx == 0 and dy==0:
        return buffer_

    with nogil:
        if dx != 0 and dy != 0:
            for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    ii = (i + dx4) % b_length
                    ii = (ii + dyw4) % b_length
                    if ii < 0:
                        ii = b_length + ii
                    new_buffer[ii]   = buffer_[i]
                    new_buffer[ii+1] = buffer_[i+1]
                    new_buffer[ii+2] = buffer_[i+2]
                    new_buffer[ii+3] = buffer_[i+3]
        else:
            if dx !=0:
                for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    ii = (i + dx4) % b_length
                    if ii < 0:
                        ii = ii + w4
                    new_buffer[ii]   = buffer_[i]
                    new_buffer[ii+1] = buffer_[i+1]
                    new_buffer[ii+2] = buffer_[i+2]
                    new_buffer[ii+3] = buffer_[i+3]
                    
            elif dy !=0:
                for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    ii = (i + dyw4) % b_length
                    if ii < 0:
                        ii = ii + b_length
                    new_buffer[ii]   = buffer_[i]
                    new_buffer[ii+1] = buffer_[i+1]
                    new_buffer[ii+2] = buffer_[i+2]
                    new_buffer[ii+3] = buffer_[i+3]
                
    return new_buffer



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, :] scroll_array24_c(unsigned char[:, :, :] rgb_array_, int dy, int dx):
    """  
    Compatible with pygame.Surface 24-bit only
    
    If the output image is flipped, transpose input array row and columns such as 
    rgb_array_ = pygame.surfarray.pixels3d(texture).transpose(1, 0, 2)
    
    Roll the value of an entire array (lateral and/or vertical)
    The roll effect can be apply to both directions at the same time.
    dy scroll texture vertically (-dy scroll up, +dy scroll down)
    dx scroll texture horizontally (-dx scroll left, +dx scroll right,
    array must be a numpy.ndarray type (w, h, 3) uint8
    This method return a scrolled numpy.ndarray
    
    :param rgb_array_: numpy.ndarray (w, h, 3) uint8 (array to scroll)
    :param dy: integer; scroll the array vertically (-dy up, +dy down)
    :param dx: integer; scroll the array horizontally (-dx left, +dx right)
    :return: return a memoryviewslice (numpy.ndarray type (w, h, 3) numpy uint8)
    """

    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))

    cdef int w, h, dim
    try:
        w, h, dim = (<object>rgb_array_).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    if dx == 0 and dy == 0:
        return rgb_array_   # return memoryslice

    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, ::1] empty_array = numpy.empty((w, h, 3), numpy.uint8)

    with nogil:
        # do both vertical and horizontal shift
        if dx != 0 and dy != 0:
            for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, h):
                    ii = (i + dx) % w
                    jj = (j + dy) % h
                    if ii < 0:
                        ii = ii + w
                    if jj < 0:
                        jj = jj + h
                    empty_array[ii, jj, 0] = rgb_array_[i, j, 0]
                    empty_array[ii, jj, 1] = rgb_array_[i, j, 1]
                    empty_array[ii, jj, 2] = rgb_array_[i, j, 2]
        else:
            if dx != 0:
                # Move horizontally
                for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        ii = (i + dx) % w
                        if ii < 0:
                            ii = ii + w
                        empty_array[ii, j, 0] = rgb_array_[i, j, 0]
                        empty_array[ii, j, 1] = rgb_array_[i, j, 1]
                        empty_array[ii, j, 2] = rgb_array_[i, j, 2]
            else:
                # Move vertically
                for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        jj = (j + dy) % h
                        if jj < 0:
                            jj = jj + h
                        empty_array[i, jj, 0] = rgb_array_[i, j, 0]
                        empty_array[i, jj, 1] = rgb_array_[i, j, 1]
                        empty_array[i, jj, 2] = rgb_array_[i, j, 2]

    return empty_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, :] scroll_array32_c(unsigned char [:, :, :] rgb_array, int dy, int dx):
    """  
    Compatible with pygame.Surface 32-bit only
    
    If the output image is flipped, transpose input array row and columns such as 
    im32 = pygame.image.load("image.jpg")
    im32 = pygame.transform.smoothscale(im32, (600, 600))
    array32 = pygame.surfarray.pixels3d(im32)
    alpha = pygame.surfarray.array_alpha(im32)
    stack_array = stack_mem(array32, alpha, True)  <-- Transpose = True
    or stack_array = numpy.ascontiguousarray(numpy.dstack((array32, alpha)).transpose(1, 0, 2))
    
    Roll the value of an entire array (lateral and vertical)
    The roll effect can be apply to both direction (vertical and horizontal) at the same time
    dy scroll texture vertically (-dy scroll up, +dy scroll down)
    dx scroll texture horizontally (-dx scroll left, +dx scroll right,
    array must be a numpy.ndarray type (w, h, 4) uint8
    This method return a scrolled numpy.ndarray
    
    :param rgb_array: numpy.ndarray (w, h, 4) uint8 (array to scroll)
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: Return a memoryslice (numpy.ndarray type (w, h, 4) numpy uint8)
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))

    cdef int w, h, dim
    try:
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    if dx == 0 and dy == 0:
        return rgb_array

    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, ::1] empty_array = numpy.empty((w, h, 4), numpy.uint8)

    with nogil:
        # do both vertical and horizontal shift
        if dx != 0 and dy != 0:
            for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, h):
                    ii = (i + dx) % w
                    jj = (j + dy) % h
                    if ii < 0:
                        ii = ii + w
                    if jj < 0:
                        jj = jj + h
                    empty_array[ii, jj, 0] = rgb_array[i, j, 0]
                    empty_array[ii, jj, 1] = rgb_array[i, j, 1]
                    empty_array[ii, jj, 2] = rgb_array[i, j, 2]
                    empty_array[ii, jj, 3] = rgb_array[i, j, 3]
        else:
            if dx != 0:
                # Move horizontally
                for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        ii = (i + dx) % w
                        if ii < 0:
                            ii = ii + w
                        empty_array[ii, j, 0] = rgb_array[i, j, 0]
                        empty_array[ii, j, 1] = rgb_array[i, j, 1]
                        empty_array[ii, j, 2] = rgb_array[i, j, 2]
                        empty_array[ii, j, 3] = rgb_array[i, j, 3]

            else:
                # Move vertically
                for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        jj = (j + dy) % h
                        if jj < 0:
                            jj = jj + h
                        empty_array[i, jj, 0] = rgb_array[i, j, 0]
                        empty_array[i, jj, 1] = rgb_array[i, j, 1]
                        empty_array[i, jj, 2] = rgb_array[i, j, 2]
                        empty_array[i, jj, 3] = rgb_array[i, j, 3]
    return empty_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_array32m_c(unsigned char [:, :, :] rgb_array,
                       unsigned char [:, :] alpha_array, int dy, int dx):
    """  
    Compatible with pygame.Surface 32-bit only
    
    e.g
    array32, alpha = scroll_array32m(array32, alpha, -1, -1)
    stack_array = stack_mem(array32, alpha, True)
    screen.blit(pygame.image.frombuffer(stack_array, (w, h), 'RGBA'), (0, 0))
    
    Roll the value of an entire array (lateral and vertical)
    The roll effect can be apply to both direction (vertical and horizontal) at the same time
    dy scroll texture vertically (-dy scroll up, +dy scroll down)
    dx scroll texture horizontally (-dx scroll left, +dx scroll right,
    array must be a numpy.ndarray type (w, h, 4) uint8
    This method return a scrolled numpy.ndarray
    
    :param alpha_array: numpy.ndarray (w, h) uint8 (texture alpha values) 
    :param rgb_array: numpy.ndarray (w, h, 4) uint8 (array to scroll)
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: a numpy.ndarray type (w, h, 4) numpy uint8
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))

    cdef int w, h, dim
    try:
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    if dx == 0 and dy == 0:
        return rgb_array

    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, ::1] new_array = numpy.empty((w, h, 3), numpy.uint8)
        unsigned char [:, :] new_alpha = numpy.empty((w, h), numpy.uint8)

    with nogil:
        # do both vertical and horizontal shift
        if dx != 0 and dy != 0:
            for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, h):
                    ii = (i + dx) % w
                    jj = (j + dy) % h
                    if ii < 0:
                        ii = ii + w
                    if jj < 0:
                        jj = jj + h
                    new_array[ii, jj, 0] = rgb_array[i, j, 0]
                    new_array[ii, jj, 1] = rgb_array[i, j, 1]
                    new_array[ii, jj, 2] = rgb_array[i, j, 2]
                    new_alpha[ii, jj] = alpha_array[i, j]
        else:
            if dx != 0:
                # Move horizontally
                for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        ii = (i + dx) % w
                        if ii < 0:
                            ii = ii + w
                        new_array[ii, j, 0] = rgb_array[i, j, 0]
                        new_array[ii, j, 1] = rgb_array[i, j, 1]
                        new_array[ii, j, 2] = rgb_array[i, j, 2]
                        new_alpha[ii, j] = alpha_array[i, j]

            else:
                # Move vertically
                for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        jj = (j + dy) % h
                        if jj < 0:
                            jj = jj + h
                        new_array[i, jj, 0] = rgb_array[i, j, 0]
                        new_array[i, jj, 1] = rgb_array[i, j, 1]
                        new_array[i, jj, 2] = rgb_array[i, j, 2]
                        new_alpha[i, jj] = alpha_array[i, j]
    return new_array, new_alpha


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef roll_array_c(array_, dx=0, dy=0):
    """
    Scroll array using numpy.roll method
    Use dy to scroll up or down (move the image of dy pixels)
    Use dx to scroll left or right (move the image of dx pixels)
    
    e.g
    Don't forget to transpose row and columns if the image output image is flipped
    array_ = pygame.surfarray.pixels3d(im).transpose(1, 0, 2)
    surface, array_ = scroll_surface(array_, -1, -1)
    screen.blit(surface, (0, 0))
        
    :param dx: int, Use dx for scrolling right or left (move the image of dx pixels)
    :param dy: int, Use dy to scroll up or down (move the image of dy pixels)
    :param array_: numpy.ndarray such as pixels3d(texture).
    This will only work on Surfaces that have 24-bit or 32-bit formats.
    Lower pixel formats cannot be referenced using this method.
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(array_, numpy.ndarray):
        raise TypeError('array, a numpy.ndarray is required (got type %s)' % type(array_))

    cdef int w, h
    try:
        w, h = (<object>array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')
    if dx != 0:
        array = numpy.roll(array_, dx, axis=1)
    if dy != 0:
        array = numpy.roll(array, dy, axis=0)
    return pygame.image.frombuffer(array, (w, h), 'RGB'), array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef roll_surface_c(surface_, dx=0, dy=0):
    """
    Scroll a pygame.Surface using numpy.roll method
       
    :param dx: int, Use dx for scrolling right or left (move the image of dx pixels)
    :param dy: int, Use dy to scroll up or down (move the image of dy pixels)
    :param surface_: pygame.Surface 24-32 bit format compatible
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(surface_, pygame.Surface):
        raise TypeError('array, a numpy.ndarray is required (got type %s)' % type(surface_))

    cdef int w, h

    try:
        w, h = surface_.get_size()

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    try:
        array = pygame.surfarray.pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            array = pygame.surfarray.array3d(surface_)
        except (pygame.error, ValueError):
            raise ValueError('Surface not compatible.')

    if dx != 0:
        array = numpy.roll(array, dx, axis=1)
    if dy != 0:
        array = numpy.roll(array, dy, axis=0)
    return pygame.image.frombuffer(array.transpose(1, 0, 2), (w, h), 'RGB'), array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_surface24_c(surface, int dy, int dx):
    """
    Scroll surface horizontally and/or vertically
    
    :param surface: pygame Surface 24, 32-bit format compatible.
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: Return a tuple (surface:Surface, array:numpy.ndarray) type (w, h, 3) numpy uint8
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(surface, pygame.Surface):
        raise TypeError('surface, a pygame.Surface is required (got type %s)' % type(surface))

    cdef int w, h, dim

    try:
        array = pixels3d(surface)
        alpha = pixels_alpha(surface)
    except (ValueError, pygame.error) as e:
        try:
            array = array3d(surface)
            alpha = array_alpha(surface)
        except (ValueError, pygame.error) as e:
            raise ValueError('\nIncompatible pixel format.')

    try:
        w, h, dim = (<object> array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    cdef:
        int i=0, j=0, ii=0, jj=0
        unsigned char [:, :, :] rgb_array = array
        unsigned char [:, :, ::1] new_array = numpy.empty((h, w, 3), dtype=uint8)

    if dx==0 and dy==0:
        return surface, array
    with nogil:
        if dx !=0 and dy != 0:
            for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        ii = (i + dx) % w
                        jj = (j + dy) % h
                        if ii < 0:
                            ii = ii + w
                        if jj < 0:
                            jj = jj + h
                        new_array[jj, ii, 0] = rgb_array[i, j, 0]
                        new_array[jj, ii, 1] = rgb_array[i, j, 1]
                        new_array[jj, ii, 2] = rgb_array[i, j, 2]
        else:
            if dx != 0:
                for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(h):
                        ii = (i + dx) % w
                        if ii < 0:
                            ii = ii + w
                        new_array[j, ii, 0], new_array[j, ii, 1], new_array[j, ii, 2] = \
                            rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
            else:
                for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(h):
                        jj = (j + dy) % h
                        if jj < 0:
                            jj = jj + h
                        new_array[j, ii, 0], new_array[j, ii, 1], new_array[j, ii, 2] = \
                            rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

    return pygame.image.frombuffer(new_array, (w, h), 'RGB'), new_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_surface32_c(surface, int dy, int dx):
    """
    Scroll surface channel alpha (lateral/vertical using optional dx, dy values)
    
    :param surface: 32 bit pygame surface only or 24-bit surface converted
    with convert_alpha() pygame method. An error will be thrown if the surface does not contains
    per-pixel transparency. 
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: Return a tuple (surface: 32 bit Surface with per-pixel info,
    array:3d array numpy.ndarray shape (w, h, 4)) 
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(surface, pygame.Surface):
        raise TypeError('surface, a pygame surface is required (got type %s)' % type(surface))

    try:
        array = pixels3d(surface)
        alpha = pixels_alpha(surface)

    except (ValueError, pygame.error) as e:
        try:
            array = pygame.surfarray.array3d(surface)
            alpha = pygame.surfarray.array_alpha(surface)
        except (ValueError, pygame.error) as e:      
            raise ValueError('\nIncompatible image pixel format.')

    cdef int w, h
    try:
        w, h = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, :] rgb_array = array
        unsigned char [:, :] alpha_array = alpha
        unsigned char [:, :, ::1] new_array = numpy.empty((w, h, 4), numpy.uint8)

    if dx==0 and dy==0:
        return surface
    
    with nogil:
        if dx !=0 and dy != 0:
            for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(0, h):
                        ii = (i + dx) % w
                        jj = (j + dy) % h
                        if ii < 0:
                            ii = ii + w
                        if jj < 0:
                            jj = jj + h
                        new_array[jj, ii, 0] = rgb_array[i, j, 0]
                        new_array[jj, ii, 1] = rgb_array[i, j, 1]
                        new_array[jj, ii, 2] = rgb_array[i, j, 2]
                        new_array[jj, ii, 3] = alpha_array[i, j]
        else:
            if dx != 0:
                for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(h):
                        ii = (i + dx) % w
                        if ii < 0:
                            ii = ii + w
                        new_array[j, ii, 0] = rgb_array[i, j, 0]
                        new_array[j, ii, 1] = rgb_array[i, j, 1]
                        new_array[j, ii, 2] = rgb_array[i, j, 2]
                        new_array[j, ii, 3] = alpha_array[i, j]
            else:
                for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for j in range(h):
                        jj = (j + dy) % h
                        if jj < 0:
                            jj = jj + h
                        new_array[jj, i, 0] = rgb_array[i, j, 1]
                        new_array[jj, i, 1] = rgb_array[i, j, 1]
                        new_array[jj, i, 2] = rgb_array[i, j, 1]
                        new_array[jj, i, 3] = alpha_array[i, j]   

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:] vfb24_c(unsigned char [:] source, unsigned char [:] target, int width, int height)nogil:
    """
    Vertically flipped buffer   

    Flip a C-buffer vertically filled with RGB values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGB otherwise a valuerror
    will be raised.
    This method is using Multiprocessing OPENMP
    e.g
    Here is a 9 pixels buffer (length = 27), pixel format RGB   

    buffer = [RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, RGB7, RGB8, RGB9]
    Equivalent 3d model would be (3x3x3):
    3d model = [RGB1 RGB2 RGB3]
               [RGB4 RGB5 RGB6]
               [RGB7 RGB8 RGB9]

    After vbf_rgb:
    output buffer = [RGB1, RGB4, RGB7, RGB2, RGB5, RGB8, RGB3, RGB6, RGB9]
    and its equivalent 3d model
    3D model = [RGB1, RGB4, RGB7]
               [RGB2, RGB5, RGB8]
               [RGB3, RGB6, RGB9]       

    :param source: 1d buffer to flip vertically (unsigned char values).
     The array length is known with (width * height * depth). The buffer represent 
     image 's pixels RGB.     
    :param target: Target buffer must have same length than source buffer)
    :param width: integer; width of the image 
    :param height: integer; height of the image
    :return: Return a vertically flipped buffer 
    """

    cdef:
        # int i, j
        cdef Py_ssize_t i, j
        int index, k 
        int w3 = width * 3

    for i in prange(0, w3, 3):
        for j in range(0, height):
            index = i + (w3 * j)
            k = (j * 3) + (i * height)
            target[k] =  <unsigned char>source[index]
            target[k + 1] =  <unsigned char>source[index + 1]
            target[k + 2] =  <unsigned char>source[index + 2]

    return target



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [::1] vfb32_c(unsigned char [:] source, unsigned char [::1] target,
                                   int width, int height)nogil:
    """
    Vertically flipped buffer
    
    Flip a C-buffer vertically filled with RGBA values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGBA otherwise a valuerror
    will be raised.
    This method is using Multiprocessing OPENMP
    e.g
    Here is a 9 pixels buffer (length = 36), pixel format RGBA
    
    buffer = [RGBA1, RGBA2, RGBA3, RGBA4, RGBA5, RGBA6, RGBA7, RGBA8, RGBA9]
    Equivalent 3d model would be (3x3x4):
    3d model = [RGBA1 RGBA2 RGBA3]
               [RGBA4 RGBA5 RGBA6]
               [RGBA7 RGBA8 RGBA9]

    After vbf_rgba:
    output buffer = [RGB1A, RGB4A, RGB7A, RGB2A, RGB5A, RGBA8, RGBA3, RGBA6, RGBA9]
    and its equivalent 3d model
    3D model = [RGBA1, RGBA4, RGBA7]
               [RGBA2, RGBA5, RGBA8]
               [RGBA3, RGBA6, RGBA9]
        
    :param source: 1d buffer to flip vertically (unsigned char values).
     The array length is known with (width * height * depth). The buffer represent 
     image 's pixels RGBA. 
     
    :param target: Target buffer must have same length than source buffer)
    :param width: integer; width of the image 
    :param height: integer; height of the image
    :return: Return a vertically flipped buffer 
    """
    cdef:
        int i, j, index, k
        int w4 = width * 4

    for i in prange(0, w4, 4):
        for j in range(0, height):
            index = i + (w4 * j)
            k = (j * 4) + (i * height)
            target[k] =  <unsigned char>source[index]
            target[k + 1] =  <unsigned char>source[index + 1]
            target[k + 2] =  <unsigned char>source[index + 2]
            target[k + 3] =  <unsigned char>source[index + 3]

    return target
