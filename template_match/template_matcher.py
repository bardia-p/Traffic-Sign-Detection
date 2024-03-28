import cv2
import os.path
import numpy

NUM_TEMPLATES = 42

class TemplateMatcher:

    my_path = os.path.abspath(os.path.dirname(__file__))

    templates = []

    for i in range(NUM_TEMPLATES):
        templates.append(cv2.imread(os.path.join(my_path,'./templates/' + str(i) + '.png')))

    def locate_sign(self, img):
        imgR, imgG, imgB = cv2.split(img)




        returned_images = []
        img_w, img_h, img_c = img.shape[::-1]

        for i in range(len(self.templates)):
            template = self.templates[i]
            templateR, templateG, templateB = cv2.split(template)
            resultB = cv2.matchTemplate(imgR, templateR, cv2.TM_SQDIFF)
            resultG = cv2.matchTemplate(imgG, templateG, cv2.TM_SQDIFF)
            resultR = cv2.matchTemplate(imgB, templateB, cv2.TM_SQDIFF)

            threshold = 0.95
            result = resultB + resultG + resultR
            loc = numpy.where(result >= 3 * threshold)
            
            w,h,c = template.shape[::-1]
            for pt in zip(*loc[::-1]):
                rec = (pt[0], pt[1], w, h)
                xmin=int(max(pt[0], 0))
                xmax=int(min(pt[0] + w, img_w))

                ymin=int(max(pt[1], 0))
                ymax=int(min(pt[1] + h, img_h))

                returned_images += [(img[xmin:xmax, ymin:ymax], rec, i)]
                break

        return returned_images
