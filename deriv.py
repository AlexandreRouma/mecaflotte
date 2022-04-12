import defs

# => Derive value <=
# f_left:       function value on the left
# f_c           function value in the center
# f_right:      function value on the right
# type_left:    node type on the left
# type_c:       node type in the center
# type_right:   node type on the right
# h:            node step size
# returns:      left to right derivative
def deriv(f_left,f_c,f_right,type_left,type_c,type_right,h):
    # Do not derive out-of-bounds nodes
    if type_c == defs.NODE_TYPE_EXTERNAL:
        v = 0

    # If the left side is out-of-bounds, do right asymetric derivative
    elif type_left == defs.NODE_TYPE_EXTERNAL:
        v = (f_right - f_c) / h

    # If the right side is out-of-bounds, do left asymetric derivative
    elif type_right == defs.NODE_TYPE_EXTERNAL:
        v = (f_c - f_left) / h

    # Otherwise, do a symetric derivative
    else:
        v  = (f_right - f_left) / (2.0 * h)

    return v