import cv2
import os.path
import numpy as np

NUM_TEMPLATES = 43

class TemplateMatcher:

    my_path = os.path.abspath(os.path.dirname(__file__))

    templates = []

    for i in range(NUM_TEMPLATES):
        templates.append(cv2.cvtColor(cv2.imread(os.path.join(my_path,'./templates/' + str(i) + '.png')), cv2.COLOR_BGR2GRAY))

    def template_match(self, img, guesses):
        '''
        Performs template matching on the image with a given set of templates.

        @param img: the img to analyze.
        @param guesses: the templates to verify.

        @returns the most likely template.
        '''
        result = []

        for _,g in guesses:
            template = self.templates[g - 1]

            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8  # Adjust this threshold according to your needs
            loc = np.where(res >= threshold)
            result.append((len(loc), g))

        return sorted(result, key=lambda x : x[0], reverse=True)
    
    def sift_match(self, img, guesses):
        '''
        Performs sift matching on the image with a given set of templates.

        @param img: the img to analyze.
        @param guesses: the templates to verify.

        @returns the most likely template.
        '''
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()

        result = []

        for _,g in guesses:
            template = self.templates[g - 1]
          
            kp2, des2 = sift.detectAndCompute(template,None)

            matches = bf.knnMatch(des1,des2,k=2)

            good_matches = []
            for j, values in enumerate(matches):
                if len(values) < 2:
                    continue

                m,n = values
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
            
            result.append((len(good_matches), g))
   
            if len(good_matches) > 10:
                # Extract matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Perform RANSAC to estimate transformation
                ransac_threshold = 5.0 
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

                match_count = np.sum(mask)

                if match_count != 0:
                    result.append((match_count, g))
    
        return sorted(result, key=lambda x : x[0], reverse=True)
