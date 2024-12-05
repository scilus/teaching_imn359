
def add_square_around_zoom(img, zoomx, zoomy, width):
    img[zoomx:zoomx + 3, zoomy:zoomy + width] = 0
    img[zoomx + width:zoomx + width + 3, zoomy:zoomy + width] = 0
    img[zoomx:zoomx + width, zoomy:zoomy + 3] = 0
    img[zoomx:zoomx + width, zoomy + width:zoomy + width + 3] = 0

    return img


def convolve_2d_info_copied_from_internet(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """
    Copied from : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

    Convolve in1 and in2 with output size determined by mode, and boundary
    conditions determined by boundary and fillvalue.

    Parameters
    ----------
    in1: array_like
        First input.
    in2: array_like
        Second input. Should have the same number of dimensions as in1.
    mode: str {‘full’, ‘valid’, ‘same’}, optional
        A string indicating the size of the output:
            - full: The output is the full discrete linear convolution of the
                inputs. (Default)
            - valid: The output consists only of those elements that do not
                rely on the zero-padding. In ‘valid’ mode, either in1 or
                in2 must be at least as large as the other in every dimension.
            - same: The output is the same size as in1, centered with respect
                to the ’full’ output.
    boundary: str {‘fill’, ‘wrap’, ‘symm’}, optional
        A flag indicating how to handle boundaries:
            - fill: pad input arrays with fillvalue. (default)
            - wrap: circular boundary conditions.
            - symm: symmetrical boundary conditions.
    fillvalue: scalar, optional
        Value to fill pad input arrays with. Default is 0.
    """
    raise NotImplementedError

