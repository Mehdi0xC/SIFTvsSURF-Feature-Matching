from unit import featureMatcher
import string
import time

config = {
	#------------------------------------------------------------
	# OPACITY OF MATCHING HIGHLIGHTS
	#------------------------------------------------------------
	"opacity": 0.25,
	#------------------------------------------------------------
	# COLOR OF MATCHING HIGHLIGHTS
	#------------------------------------------------------------	
	"matching_color": (0,255,0),
	#------------------------------------------------------------	
	# THRESHOLD FOR DETERMINING GOOD MATCHES, LOWER = MORE ACCURATE (KINDA PRECISION/RECALL TRADEOFF)
	#------------------------------------------------------------	
	"matching_threshold": 0.8,
	#------------------------------------------------------------
}	

if __name__ == '__main__':
	methods = ["sift", "surf"]
	report_file = open("../REPORT.txt","a")
	for method in methods:
		t = time.time()
		for imageIdx in range(5):
			for templateIdx in range(5):
				image_file_name = "img" + str(imageIdx) + "/image" + str(imageIdx) + ".png"
				template_file_name = "img" + str(imageIdx) + "/template" + str(templateIdx)+ ".png"	
				featureMatcher(
					image_file_name = image_file_name, 
					template_file_name = template_file_name, 
					method = method, 
					config = config, 
					imageIdx = imageIdx, 
					templateIdx = templateIdx)
		log = "Time elapsed for " + method.upper() + " method:" + str(time.time() - t) + "\n"	
		report_file.write(log)
	report_file.close()
