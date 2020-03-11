# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, dstack, full, ones, \
        asarray, ascontiguousarray
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")
    raise SystemExit

# OPENCV IS REQUIRED
try:
    import cv2
except ImportError:
    print("\n<cv2> library is missing on your system."
          "\nTry: \n   C:\\pip install opencv-python on a window command prompt.")
    raise SystemExit

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

try:
    import Scroll
    from Scroll import scroll_array24, scroll_array32, scroll_array32m, \
        stack_object, stack_mem, transpose24, transpose32, roll_surface, roll_array, \
        scroll_surface24, scroll_surface32, unstack_object, unstack_buffer, \
        stack_buffer, vfb24, vfb32, scroll_buffer24, scroll_buffer32
except ImportError:
    print("\nScroll.pyx is missing.\n"
          "Try: C:>python setup_scroll.py build_ext --inplace")

import timeit

N = 1000

if __name__ == '__main__':

    """
    a = numpy.array([x for x in range(8 * 8 * 3)], numpy.uint8)  # create a contiguous array 8x8x3 pixels
    aa = a.reshape(8, 8, 3) # --> create a 3d array of RGB values
    print(aa.shape, aa.flags, aa.strides)
    print(aa)
    b = aa.transpose(1, 0, 2)
    print(b.shape)
    print(b.flags)
    print(b.strides)
    print('b', b.flatten())

    empty = numpy.empty(8*8*3, numpy.uint8)
    d = numpy.asarray(Scroll.vfb_rgb(a, empty, 8, 8))
    print('d', d)

    c = aa.T
    print(c.shape)
    print(c.flags)
    print(c.strides)
    print(c)

    screen = pygame.display.set_mode((8*3, 8))
    screen.blit(pygame.image.frombuffer(c, (8, 8), 'RGB'), (0, 0))
    pygame.display.flip()
    """

    im = pygame.image.load("sand24.jpg")
    im = pygame.transform.smoothscale(im, (500, 500))

    w, h = im.get_size()
    screen = pygame.display.set_mode((w, h * 2))
    array_ = pygame.surfarray.pixels3d(im).transpose(1, 0, 2)

    im32 = pygame.image.load("sand32.png")
    im32 = pygame.transform.smoothscale(im32, (w, h))

    # print("PYGAME SCROLL 24 :", timeit.timeit("pygame.Surface.scroll(im, 1, 1)",
    #                                           "from __main__ import pygame, im", number=N) / N)
    # print("PYGAME SCROLL 32 :", timeit.timeit("pygame.Surface.scroll(im32, 1, 1)",
    #                                           "from __main__ import pygame, im32", number=N) / N)
    dx = 1
    dy = -1
    DEMO = True
    while DEMO:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            DEMO = False
        pygame.Surface.scroll(im32, 1, 1)

        screen.blit(im32, (0, 0))
        pygame.display.flip()

    array32 = pygame.surfarray.pixels3d(im32)
    alpha = pygame.surfarray.array_alpha(im32)
    # ---------------------- stack mem -------------------------------------
    stack_array = stack_mem(array32, alpha, False)
    print("STACK ARRAY (mem) : ", timeit.repeat(
        "stack_mem(array32, alpha, False)",
        "from __main__ import stack_mem, array32, alpha", repeat=3, number=N))
    a = stack_mem(array32, alpha, False)
    print("STACK ARRAY (mem) output flags : ", numpy.array(a).flags)
    print("STACK ARRAY (mem) type :", type(a))

    print("STACK ARRAY (mem) flipped: ", timeit.repeat(
        "stack_mem(array32, alpha, True)",
        "from __main__ import stack_mem, array32, alpha", repeat=3, number=N))
    # ----------------------------------------------------------------------

    # ---------------------- unstack array object -----------------------------------
    array32_unstack, alpha_unstack = unstack_object(stack_array)
    assert numpy.array_equal(array32, array32_unstack), "RGB Arrays are not identical"
    assert numpy.array_equal(alpha, alpha_unstack), "RGB Arrays are not identical"
    print("UNSTACK ARRAY (obj) : ", timeit.timeit("unstack_object(stack_array)",
                                                  "from __main__ import unstack_object, stack_array", number=N) / N)
    print("UNSTACK ARRAY (obj) : ", timeit.repeat(
        "unstack_object(stack_array)",
        "from __main__ import unstack_object, stack_array", repeat=3, number=N))
    a, b = unstack_object(stack_array)
    print("UNSTACK ARRAY (obj) output flags : ", a.flags)
    print("UNSTACK ARRAY (obj) output flags : ", b.flags)
    print("UNSTACK ARRAY (obj) type :", type(a), type(b))
    # ----------------------------------------------------------------------

    # ---------------------- unstack buffer -----------------------------------
    buf = numpy.frombuffer(im32.get_view('2'), numpy.uint8)
    rgb_buf, alpha_buf = unstack_buffer(buf, w, h)
    print("UNSTACK BUFFER: ", timeit.repeat("unstack_buffer(buf, w, h)",
                                            "from __main__ import unstack_buffer, buf, w, h", repeat=3, number=N))

    a, b = unstack_buffer(buf, w, h)
    print("UNSTACK ARRAY (obj) output flags : ", numpy.array(a).flags)
    print("UNSTACK ARRAY (obj) output flags : ", numpy.array(b).flags)
    print("UNSTACK ARRAY (obj) type :", type(a), type(b))

    array1 = numpy.array(rgb_buf, dtype=uint8)

    array2 = numpy.array(im32.get_view('3'), dtype=uint8).transpose(1, 0, 2)
    array2 = array2.flatten(order='C')

    screen.blit(pygame.image.frombuffer(array1, (w, h), 'RGB'), (0, 0))
    screen.blit(pygame.image.frombuffer(array2, (w, h), 'RGB'), (0, h))
    pygame.display.flip()

    print("ARRAY 1:\n", array1)
    print("ARRAY 2:\n", array2)
    if not numpy.array_equal(array1, array2):
        raise ValueError("RGB Arrays are not identical")

    alpha1 = numpy.frombuffer(alpha_buf, uint8)
    alpha2 = numpy.array(im32.get_view('a'), uint8).transpose(1, 0)
    alpha2 = alpha2.flatten(order='C')

    print("ALPHA 1:\n", alpha1, len(alpha1))
    print("ALPHA 2:\n", alpha2, len(alpha2))

    for r in range(len(alpha1)):
        if alpha1[r] != alpha2[r]:
            print(r, alpha1[r], alpha2[r])
    if not numpy.array_equal(alpha1, alpha2):
        raise ValueError("RGB Arrays are not identical")

    buf = im32.get_view('2')
    assert buf.length == w * h * 4, "Incorrect buffer length %s %s " % (buf.length, w * h * 4)
    rgb_buf, alpha_buf = unstack_buffer(numpy.frombuffer(buf, numpy.uint8), w, h)
    assert len(rgb_buf) == w * h * 3, "Incorrect buffer length %s %s " % (len(rgb_buf.shape), w * h * 3)
    assert len(alpha_buf) == w * h, "Incorrect buffer length %s %s " % (len(alpha_buf), w * h)
    # ---------------------- unstack buffer -----------------------------------

    # ---------------------- stack buffer -----------------------------------
    buf_ = stack_buffer(rgb_buf, alpha_buf, w, h, False)
    assert len(buf_) == w * h * 4, "Incorrect buffer length %s %s " % (len(buf_), w * h * 4)

    print("STACK BUFFERS : ", timeit.repeat(
        "stack_buffer(rgb_buf, alpha_buf, w, h, False)",
        "from __main__ import stack_buffer, rgb_buf, alpha_buf, w, h", repeat=3, number=N))

    print("STACK BUFFERS (flipped): ", timeit.repeat(
        "stack_buffer(rgb_buf, alpha_buf, w, h, True)",
        "from __main__ import stack_buffer, rgb_buf, alpha_buf, w, h", repeat=3, number=N))

    a = stack_buffer(rgb_buf, alpha_buf, w, h, True)
    print("STACK BUFFER output flags : ", numpy.array(a).flags)
    print("STACK BUFFER type :", type(a))
    # ---------------------- stack buffer -----------------------------------

    stack_array = stack_mem(array32, alpha, True)

    N = 100
    print("SCROLL ARRAY 24 :", timeit.timeit("scroll_array24(array_, 1, 1)",
                                             "from __main__ import scroll_array24, array_", number=N) / N)

    print("SCROLL ARRAY 32 :", timeit.timeit("scroll_array32(stack_array, 1, 1)",
                                             "from __main__ import scroll_array32, stack_array", number=N) / N)

    print("SCROLL ARRAY 32m :", timeit.timeit("scroll_array32m(array32, alpha, 1, 1)",
                                              "from __main__ import scroll_array32m, array32, alpha", number=N) / N)

    print("ROLL SURFACE :", timeit.timeit("roll_surface(im, 1, 1)",
                                          "from __main__ import roll_surface, im", number=N) / N)

    print("ROLL ARRAY :", timeit.timeit("roll_array(array_, 1, 1)",
                                        "from __main__ import roll_array, array_", number=N) / N)

    print("SCROLL SURFACE 24 :", timeit.timeit("scroll_surface24(im, 1, 1)",
                                               "from __main__ import scroll_surface24, im", number=N) / N)

    print("SCROLL SURFACE 32 :", timeit.timeit("scroll_surface32(im32, 1, 1)",
                                               "from __main__ import scroll_surface32, im32", number=N) / N)

    array = numpy.array(im32.get_view('3'), dtype=uint8)
    array = array.flatten(order='C')
    print("SCROLL buffer 24 :", timeit.timeit("scroll_buffer24(array, w, h, 1, 1)",
                                              "from __main__ import scroll_buffer24, w, h, array", number=N) / N)

    dx = 1
    dy = -1
    DEMO = True
    while DEMO:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            DEMO = False
        array = scroll_buffer24(array, w, h, dy, dx)
        surface = pygame.image.frombuffer(array, (w, h), 'RGB')
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    buff = numpy.frombuffer(im32.get_view('2'), dtype=uint8)
    print(buff.shape, w, h, w * h * 4)
    print("SCROLL buffer 32 :", timeit.timeit("scroll_buffer32(buff, w, h, 1, 1)",
                                              "from __main__ import scroll_buffer32, w, h, buff", number=N) / N)
    dx = 1
    dy = -1
    DEMO = True
    while DEMO:
        screen.fill((255, 0, 0, 0))
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            DEMO = False
        buff = scroll_buffer32(buff, w, h, dy, dx)
        surface = pygame.image.frombuffer(buff, (w, h), 'RGBA')
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    print('\nTranspose')
    print(timeit.timeit("transpose24(array_)",
                        "from __main__ import transpose24, array_", number=N) / N)

    print(timeit.timeit("transpose32(stack_array)",
                        "from __main__ import transpose32, stack_array", number=N) / N)
    print('\nTranspose buffer')
    empty = numpy.empty(w * h * 3, numpy.uint8)
    array1_ = numpy.array(im.get_view('3'), numpy.uint8).flatten()
    print(timeit.timeit("vfb24(array1_, empty, w, h)",
                        "from __main__ import vfb24, stack_array, array1_, w, h, empty", number=N) / N)
    array1_ = numpy.frombuffer(im32.get_view('2'), numpy.uint8)
    empty = numpy.empty(w * h * 4, numpy.uint8)
    print(timeit.timeit("vfb32(array1_, empty, w, h)",
                        "from __main__ import vfb32, stack_array, array1_, w, h, empty", number=N) / N)
    stack_array = stack_mem(array32, alpha, False)
    stack_array = numpy.asarray(stack_array)
    print(timeit.timeit("stack_array.transpose(1, 0, 2)",
                        "from __main__ import stack_array", number=N) / N)

    print('\nMake_array')
    print(timeit.timeit("stack_mem(array32, alpha, True)",
                        "from __main__ import stack_mem, array32, alpha", number=N) / N)
    print(timeit.timeit("numpy.ascontiguousarray(numpy.dstack((array32, alpha)).transpose(1, 0, 2))",
                        "from __main__ import numpy, array32, alpha", number=N) / N)

    stack_array = numpy.ascontiguousarray(numpy.dstack((array32, alpha)).transpose(1, 0, 2))

    del array32, alpha

    i = 0
    j = 255
    STOP_DEMO = True
    while STOP_DEMO:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        # screen.fill((255, 0, 0, 0))

        # test scroll_array24
        # array_ = scroll_array24(array_, 0, 1)
        # screen.blit(pygame.image.frombuffer(array_, (w, h), 'RGB'), (0, 0))

        # test scroll_array32
        # stack_array = scroll_array32(stack_array, 1, 1)
        # screen.blit(pygame.image.frombuffer(stack_array, (w, h), 'RGBA'), (0, 0))

        # test scroll_array32m
        # array32, alpha = scroll_array32m(array32, alpha, -1, -1)
        # stack_array = stack_mem(array32, alpha, True)
        # screen.blit(pygame.image.frombuffer(stack_array, (w, h), 'RGBA'), (0, 0))

        # surface, array_ = scroll_surface(array_, -1, -1)
        # screen.blit(surface, (0, 0))

        # im, array_ = scroll_surface24(im, 1, 0)
        # screen.blit(im, (0, 0))

        im32 = scroll_surface32(im32, 0, 0)
        screen.blit(im32, (0, 0))

        pygame.display.flip()

        if keys[pygame.K_ESCAPE]:
            STOP_DEMO = False

        if keys[pygame.K_F8]:
            pygame.image.save(screen, 'Screendump' + str(i) + '.png')

        i += 1
        j -= 1

        if j < 0:
            j = 255
