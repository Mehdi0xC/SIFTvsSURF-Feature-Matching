import cv2
import numpy as np
import matplotlib.pyplot as plt

def featureMatcher(image_file_name, template_file_name, method, config, imageIdx, templateIdx):

	# STEP 1: LOADING THE TEMPLATE AND MAIN IMAGES
	print('../images/' + image_file_name)
	template = cv2.imread('../images/' + template_file_name ,1)    
	image = cv2.imread('../images/' + image_file_name ,1) 
	overlay = image.copy()


	# STEP 2: EXTRACTING THE FEATURES
	if method == "sift":
		algorithm = cv2.xfeatures2d.SIFT_create()
	elif method == "surf":
		algorithm = cv2.xfeatures2d.SURF_create(400)
	kp1, des1 = algorithm.detectAndCompute(template,None)
	kp2, des2 = algorithm.detectAndCompute(image,None)

	# STEP 3: MATCHING 
	detector = cv2.BFMatcher()
	matches = detector.knnMatch(des1,des2,k=2)

	# STEP 4: SELECTING APPROPRIATE MATCHES  
	selected_matches = []
	for match1,match2 in matches:
	    if match1.distance < config["matching_threshold"]*match2.distance:
	        selected_matches.append(match1)

	# STEP 5: HIGHLIGHTING TRANSPARENT CIRCLES FOR BEST MATCHES ON THE MAIN PICTURE
	points = np.float32([ kp2[m.trainIdx].pt for m in selected_matches]).reshape(-1,2)
	for p in points:
		cv2.circle(overlay, (p[0], p[1]), 25, config["matching_color"], -1);
	cv2.addWeighted(overlay, config["opacity"], image, 1 - config["opacity"], 0, image)

	# STEP 6: DRAWING COLORED MATCHING LINES
	result = cv2.drawMatchesKnn(template,kp1,image,kp2,[selected_matches],None,flags=2)
	result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

	# STEP 7: SAVING THE RESULT
	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(111)
	ax.imshow(result)
	plt.savefig("../results_" + method + "/" + str(imageIdx) + "_" + str(templateIdx) + '.png')
	plt.close()

if __name__ == "__main__":
	default_config = {
	# OPACITY OF MATCHING HIGHLIGHTS
	"opacity" : 0.25,
	# COLOR OF MATCHING HIGHLIGHTS
	"matching_color" : (0,255,255),
	# THRESHOLD FOR DETERMINING GOOD MATCHES, LOWER = MORE ACCURATE (KINDA PRECISION/RECALL TRADEOFF)
	"matching_threshold": 0.75,
	}	
	featureMatcher("image1.png", "template1.png", method = "sift", config = default_config, imageIdx=99, templateIdx=99)




