import cv2
import os.path

class SignDetector:

    my_path = os.path.abspath(os.path.dirname(__file__))

    # Get Triangle Stuff
    triangle = cv2.imread(os.path.join(my_path,'./shapes/triangle.png'), cv2.IMREAD_GRAYSCALE)
    ret_tri, thresh_tri = cv2.threshold(triangle, 127, 255, 0)
    contours_tri, hierarchy_tri = cv2.findContours(thresh_tri, 2, 1)
    tri_shape = contours_tri[1]
    tri_thresh = 0.1

    # Get Circle Stuff
    circle = cv2.imread(os.path.join(my_path,'./shapes/circle.png'), cv2.IMREAD_GRAYSCALE)
    ret_cir, thresh_cir = cv2.threshold(circle, 127, 255, 0)
    contours_cir, hierarchy_cir = cv2.findContours(thresh_cir, 2, 1)
    cir_shape = contours_cir[1]
    cir_thresh = 0.1

    # Get Square Stuff
    square = cv2.imread(os.path.join(my_path,'./shapes/square.png'), cv2.IMREAD_GRAYSCALE)
    ret_sqr, thresh_sqr = cv2.threshold(square, 127, 255, 0)
    contours_sqr, hierarchy_sqr = cv2.findContours(thresh_sqr, 2, 1)
    sqr_shape = contours_sqr[1]
    sqr_thresh = 0.1

    # Get Hexagon Stuff
    hexa = cv2.imread(os.path.join(my_path,'./shapes/hexagon.png'), cv2.IMREAD_GRAYSCALE)
    ret_hex, thresh_hex = cv2.threshold(hexa, 127, 255, 0)
    contours_hex, hierarchy_hex = cv2.findContours(thresh_hex, 2, 1)
    hex_shape = contours_hex[1]
    hex_thresh = 0.1


    def find_signs(self, img):
        can = cv2.Canny(img, 127, 255)
        ret, thresh = cv2.threshold(can, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = []
        rectangles = []
        im2 = img.copy()

        for c in contours:
            if len(c) > 100:
                x, y, w, h = cv2.boundingRect(c)
                ratio = w / h if w > h else h / w
                if ratio < 1.5:
                    match_triangle = cv2.matchShapes(c, self.tri_shape, 1, 0.0)
                    match_circle = cv2.matchShapes(c, self.cir_shape, 1, 0.0)
                    match_square = cv2.matchShapes(c, self.sqr_shape, 1, 0.0)
                    match_hexagon = cv2.matchShapes(c, self.hex_shape, 1, 0.0)
                    matches = []
                    if match_triangle < self.tri_thresh:
                        matches += ['triangle: ' + str(match_triangle)]
                    if match_square < self.sqr_thresh:
                        matches += ['square' + str(match_square)]
                    if match_circle <= self.cir_thresh:
                        matches += ['circle' + str(match_circle)]
                    if match_hexagon <= self.hex_thresh:
                        matches += ['hexagon' + str(match_hexagon)]
                    if len(matches) > 0:
                        new_contours += [c]
                        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.drawContours(im2, [c], -1, (0, 127, 0), 3)
                        found_same = False
                        cur_rect = [x, y, x + w, y + h]
                        for rec in rectangles:
                            if self.check_collide(cur_rect, [rec[0], rec[1], rec[0] + rec[2], rec[1] + rec[3]]):
                                found_same = True
                                break

                        if not found_same:
                            rectangles += [(x, y, w, h)]
                            # print(str((x, y, w, h)) + ' from ' + str(matches))

        returned_images = []
        for rec in rectangles:

            x_bot = int(max(0, rec[1] - rec[3] / 20))
            x_far = int(min(img.shape[0], rec[1] + (1.05 * rec[3])))
            y_bot = int(max(0, rec[0] - rec[2] / 20))
            y_far = int(min(img.shape[1], rec[0] + (1.05 * rec[2])))
            returned_images += [(img[x_bot:x_far, y_bot:y_far], rec)]

        return returned_images
    
    def check_collide(self, R1, R2):
      '''
      Checks to see if two rectangles collide or not.

      @param R1 rectangle 1
      @param R2 rectangle 2

      @returns true if they collide 
      '''
      if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
         return False
      else:
         return True
