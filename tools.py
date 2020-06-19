import numpy as np
import cv2


def crop(img, corners, make_square=True):

    cnt = np.array([corners[0],corners[1],corners[2],corners[3] ])
    rect = cv2.minAreaRect(cnt)
    center, size, theta = rect

    # Angle correction
    if theta < -45:
        theta += 90
    
    rect = (center, size, theta)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get width and height of the rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = np.float32([corners[0],corners[1],corners[2],corners[3]])
    dst_pts = np.float32([[0,0],[width,0],[0,height],[width,height]])

    #perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv  = cv2.getPerspectiveTransform(dst_pts, src_pts);

    # warping so as to obtain straight sudoku
    warped = cv2.warpPerspective(img, M, (width, height))

    # Making it square
    if make_square is True:
        try:
            warped = cv2.resize(warped, (max(width, height), max(width, height)), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(e)

    transformation_data = {
        'matrix' : M,
        'original_shape': (height, width),
        'matrix_inv' : M_inv
    }

    return warped, transformation_data


def unwarp_image(img, transformation_matrix, original_shape=None , output_shape=(480, 640)):
    warped_img = img

    if original_shape is not None:
        if original_shape[0]>0 and original_shape[1]>0:
            warped_img = cv2.resize(warped_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)


    unwarped_img = cv2.warpPerspective(warped_img,transformation_matrix, output_shape)
    return unwarped_img


def blend_with_original(face_img, overlay_img):
    # Let's find a mask covering all the non-black (foreground) pixels
    # NB: We need to do this on grayscale version of the image
    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]

    # Let's shrink and blur it a little to make the transitions smoother...
    overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))

    # And the inverse mask, that covers all the black (background) pixels
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def crop_minAreaRect(src, rect):
    # Get center, size, and angle from rect
    center, size, theta = rect

    # Angle correction
    if theta < -45:
        theta += 90

    # Convert to int 
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))
    out = cv2.getRectSubPix(dst, size, center)
    return out


def make_square(image, goal_dimension=28, border=2):
    height, width = image.shape[0], image.shape[1]
    dim_max = max(height, width)

    proportion = goal_dimension/dim_max

    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=BLACK)
    final = np.zeros((goal_dimension, goal_dimension), dtype=np.int)
    
    resized = cv2.resize(constant, (int(round(width*proportion)), int(round(height*proportion))), interpolation=cv2.INTER_AREA)
    x_offset=(goal_dimension-resized.shape[1])//2
    y_offset=(goal_dimension-resized.shape[0])//2

    final[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    return np.uint8(final)


class Singleton:
    """
    The purpose of the singleton class is to control object creation, limiting the number of objects to only one. 
    The singleton allows only one entry point to create the new instance of the class.
    Singletons are often useful where we have to control the resources,from a single entry point .
    
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)

