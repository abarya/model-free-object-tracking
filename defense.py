''' This is the implementation of the paper-"In Defense of Color-based Model-free Tracking" by H.Possegger
	If you wish to use this code-then you may edit a couple of lines in the beginning of the main function 
	and some lines near line 310 as per how you get the raw frames for object-tracking.
	If you have any suggestions/edits for this then do let me know.
	Created by Abhishek Arya
'''

# Size of surroundings has been kept twice the object size
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import math
import copy
import time
import os
from numpy import array
refPt = []
cropping = False
list_refpt=[]
bin=10 # no. of bins per channel
lamda=0.5 #weight parameter for the combined model
update_para=0.1
lamda_v=0.5
sigma_square=1 # other values can also be chosen  

# function for labelling object 
def click_and_crop(event, x, y, flags, param):

	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 		
		# draw a rectangle around the region of interest
		cv2.rectangle(img_copy, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", img_copy)
		cv2.waitKey(0)

def mask_bg(object_window,img) :
	''' This function outputs the surrounding pixels
	    Basically, image of background with masked target object'''
	global h_img,w_img
	x,y,w,h=object_window
	h_bg=h*2
	w_bg=2*w
	h_=0.5*h
	w_=0.5*w
	x_bg=max(x-(w_),0)
	y_bg=max(y-(h_),0)
	x_bg1=min(x_bg+w_bg,w_img-1)
	y_bg1=min(y_bg+h_bg,h_img-1)
	img[y:y+h,x:x+w]=0
	#print object_window
	#print x_bg,y_bg,x_bg1,y_bg1,img.shape
	bg_img=img[y_bg:y_bg1,x_bg:x_bg1]
	#cv2.imshow("masked_background",bg_img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return bg_img

def likelihood_map((hist1,hist2),image) :
	'''This functon generates the likelihood map based on either obj-surr/dist model
	   input: histogram of object,surr/distractors and input image
	   output:likelihood map, an image(each pixel value=corresponding probability)'''
	global h_img,w_img
	H=hist1+hist2 # histogram of I(O U S) or I(O U D)U is union 
	image_10=image/25.6 # histogram has 10 bins
	image_10=image_10.astype('uint8')
	# creating a likelihood image acc. to obj-surr or obj-distractor model
	a=image_10[:,:,0]
	a=a.ravel()
	b=image_10[:,:,1]
	b=b.ravel()
	c_=image_10[:,:,2]
	c_=c_.ravel()
	H_obj=hist1[a,b,c_] # image with pixel value=bin count of the pixel value at the same location in original image
	H_img=H[a,b,c_]
	Prob1=np.zeros((h_img*w_img,),dtype='float')
	H_obj=H_obj.astype('float')
	H_img=H_img.astype('float')
	mask=H_img==0
	#print mask,"check itjhjnkjkjkjk"
	Prob1[~mask]=np.divide(H_obj[~mask],H_img[~mask])
	Prob1[mask]=0.5
	Prob1=Prob1.reshape((h_img,w_img))
	Prob2=(Prob1)*255
	Prob2=Prob2.astype('uint8')
	likemap=cv2.applyColorMap(Prob2, cv2.COLORMAP_JET)
	return likemap,Prob1

def prob_obj(hist1,hist2) :
	'''This function creates a look-up table that contains the probability associated with the possible bin values.
	   In our case total bins=10*10*10. This thing will be computed for each frame. Then, when we need to localize the object
	   in the next frame, it will be used.
	   Input: histogram of object,surr/distractors
	   output:array of size 10x10x10 containing probability values for the corresponding bin. This array is called object model.'''
	prob=np.zeros((10,10,10),dtype='float32')

	for i in range(10) :
		for j in range(10) :
			for k in range(10) :
				if hist1[i][j][k]>0 or hist2[i][j][k] >0 :
					prob[i][j][k]= hist1[i][j][k]/(hist1[i][j][k]+hist2[i][j][k])
				else :
					prob[i][j][k]=0.5

	return prob

def label(image) :
	'''This function along with click_and_crop() helps in labelling object and background.
	   Input : Input image
	   Output: selected region of interest(either obj or distractor)'''
	global refPt,cropping,img_copy,clone,list_refpt
	#image1=copy.deepcopy(image)
	#clone = image1.copy()
	#img_copy=image1.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	print "Label the object"
	print "After making a bounding box, press 'c' "
	print "if you wish to select the object again, press 'r' "

	while True:
	# display the image and wait for a keypress
		cv2.imshow("image", img_copy)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()
			img_copy=image.copy()
	 
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break
 	
 	# if there are two reference points, then crop the region of interest
	# from the image and display it
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		cv2.imshow("ROI", roi)
		print "press any key"
		cv2.waitKey(0)

	cv2.destroyAllWindows() # close all open windows
	obj_img=roi  # roi containing the object
	list_refpt.append(refPt)
	print list_refpt,"check_list"
	return obj_img

def update_model(Prob_comb,Prob_comb_new) :
	'''This function is used to update the object model to adapt to the changing conditions.
	   Input : current_object model; object model computed using prob_obj() for the current frame
	   Output: new_object model'''
	global update_para
	Prob_comb=Prob_comb_new*update_para + (1-update_para)*Prob_comb
	return Prob_comb

def vote_score(prob_comb,obj_candidate) :
	'''vote score is computed based on the current object model to localise the object in the search region
	    Input : object_model(combined probability),object candidate
	    Output: vote_score'''
	image_10=obj_candidate/25.6 # histogram has 10 bins
	image_10=image_10.astype('uint8')
	# creating a likelihood image acc. to obj-surr or obj-distractor model
	a=image_10[:,:,0]
	a=a.ravel()
	b=image_10[:,:,1]
	b=b.ravel()
	c_=image_10[:,:,2]
	c_=c_.ravel()
	prob_candidate=prob_comb[a,b,c_]
	score=np.sum(prob_candidate)
	return score

def distance_score(cand_grid,c_x,c_y) :
	'''This is computed so as to penalize the large object movements in the successive frames
	   Input: Array containing pixel locations of the current candidate,center of current_object_window'''
	cand_grid_x=cand_grid[1]
	cand_grid_y=cand_grid[0]
	cand_grid_x=cand_grid_x-c_x
	cand_grid_y=cand_grid_y-c_y
	cand_grid_x=np.square(cand_grid_x)
	cand_grid_y=np.square(cand_grid_y)
	cand_grid_added=(cand_grid_x+cand_grid_y)*(-1/(2*sigma_square))
	exp=np.exp(cand_grid_added)
	score=np.sum(exp)
	
	return score

if __name__ == "__main__":
	argument=sys.argv
	cap=cv2.VideoCapture("mstrack.mp4")
	var=1
	if (len(argument)<2) :
		print "\n \n provide an image as input\n\n"
		if var==1 :
			folder_name="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/vot14/car"
			image=cv2.imread(folder_name +"/00000001.jpg")
			newpath="/media/arya/54E4C473E4C458BE/Users/hp/Documents/object-tracking/result_trellis"
			if not os.path.exists(newpath) :
				os.makedirs(newpath)
		else :	
			ret,image=cap.read()

	if (len(argument)==2):	
		image=cv2.imread(str(argument[1])) # complete image of the scene

	h_img,w_img,c=image.shape
	clone=image.copy()
	img_copy=image.copy()
	obj_img=label(image)  # for labelling object pixels
	dist_img_list=[]

	print "......................labelling distractors..................."

	while(1) :
		font = cv2.FONT_HERSHEY_SIMPLEX
		img=np.zeros((300,300),dtype='uint8')
		cv2.putText(img,"if you are done labelling distractors,",(10,100), font, 0.5,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(img,"type 'o' at the image window",(10,120), font, 0.5,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow('hello',img)
		k=cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()
		print "if you are done labelling distractors, type 'o' at the image window "

		if(k==ord("o")) :
			print "out"
			break

		cv2.destroyAllWindows()
		clone=image.copy()
		img_copy=image.copy()
		dist_img=label(image)
		dist_img_list.append(dist_img)

	img_copy=image.copy()
	i=0
	print len(dist_img_list)
	while(i<len(list_refpt)) :
		print i
		if(i>0) :
			frame=cv2.rectangle(img_copy, list_refpt[i][0], list_refpt[i][1], (0, 255, 0), 1)

		else :
			frame=cv2.rectangle(img_copy, list_refpt[i][0], list_refpt[i][1], (0, 0, 255), 1)

		i=i+1

	cv2.imshow("frame",frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
	#############
	# creating likelihood maps based on object-surr and object-dist model
	#############
	h,w,c=obj_img.shape	
	image1=copy.deepcopy(image)
	object_window=(list_refpt[0][0][0],list_refpt[0][0][1],w,h)
	print object_window
	bg_img=mask_bg(object_window,image1) # getting background pixels
	# computing the histograms for object and background
	hist_obj = cv2.calcHist([obj_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
	hist_bg  = cv2.calcHist([bg_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
	hist_D=0
	for i in range(len(dist_img_list)) :
		print "i",i
		hist_dist=cv2.calcHist([dist_img_list[i]],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		hist_D=hist_D+hist_dist

	# removing the effect of the pixels of the object, as object pixels had (0,0,0) pixel value in bg_img
	hist_bg[0][0][0]=hist_bg[0][0][0]- np.sum(hist_obj) 
	color_map_surr,Prob_surr=likelihood_map((hist_obj,hist_bg),image) # call likelihood function for obj-surr model
	color_map_dist,Prob_dist=likelihood_map((hist_obj,hist_D),image)# call likelihood function for obj-dist model

	#final_map = color_map_dist.astype('float32')*lamda + color_map_surr.astype('float32')*(1-lamda)
	Prob_comb=Prob_surr*(1-lamda)+Prob_dist*lamda
	final_map=Prob_comb*255
	color_map_final=final_map.astype('uint8')
	color_map_final=cv2.applyColorMap(color_map_final, cv2.COLORMAP_JET)
	cv2.imshow("obj-surr model",color_map_surr)
	cv2.imshow("distractor-aware model",color_map_dist)
	cv2.imshow("combined",color_map_final)
	cv2.imshow("original_frame",frame)
	cv2.waitKey(0)	
	cv2.destroyAllWindows()
	
	#prob as per bin no.

	prob_S=prob_obj(hist_obj,hist_bg)
	prob_D=prob_obj(hist_obj,hist_D)
	prob_comb=prob_D*lamda+prob_S*(1-lamda)

	path, dirs, files = os.walk(folder_name).next()
	num_images = len(files)

	filenames=["%08d" % number for number in range(num_images)]

	i=2
	while(i<num_images+1) :
		t=time.time()
		# var variable is 1 when reading from a image files..if reading from a video set var=0 
		if var==1 :
			new_img=cv2.imread(folder_name+"/"+filenames[i]+".jpg")
		else :	
			ret,new_img=cap.read()

		if(new_img==None) :
			break	

		new_frame=copy.deepcopy(new_img)
		x,y,w,h=object_window  
		rectangle_image=copy.deepcopy(new_img) # copy of image, will be used during scale estimation

		# setting variables for search window.. Also putting constraints like
		# variables shouldn't go beyond width/height of image

		# leftmost and topmost corner
		start_x=int(max(0,x-w)) 
		start_y=int(max(0,y-h))
		# rightmost and bottommost corner
		end_x=int(min(x+w,w_img-w))
		end_y=int(min(y+h,h_img-h))
		# center of object-window
		c_x=int(x+(w/2)) 
		c_y=int(y+(h/2))
		score_max=0
		y0=start_y
		x0=start_x
		list_score=np.zeros((end_y-y0,end_x-x0),dtype='float') # array for containing vote-scores of all the candidates, will be used for updating distractors
		# score for all the candidates is computed and simultaneously filing the above array
		# Also, we calculate the obj-cand with max score 
		while (start_x<end_x) :

			start_y=int(max(y-h,0))
			while(start_y<end_y) :
				cand_grid=np.mgrid[start_y:start_y+h,start_x:start_x+w]
				obj_cand =new_img[start_y:start_y+h,start_x:start_x+w,:]
				score1=distance_score(cand_grid,c_x,c_y) # distance-score for the combined object candidate
				score2=vote_score(prob_comb,obj_cand)  # vote-score for the combined object candidate
				score =score1*score2
				frame_next=copy.deepcopy(frame)
				list_score[start_y-y0][start_x-x0]=score2	
				if score>score_max :
					score_max=score
					score2_objcand=score2
					point1=(start_x,start_y) # contains the corner pt for the object-cand with max score
				start_y=start_y+8 # sampling after 8 pixels
			start_x=start_x+8	

		point2=(point1[0]+w,point1[1]+h) # diagonally opposite point to point1
		#cv2.rectangle(new_img,point1,point2,(0, 255, 0), 2)
		obj_img=new_frame[point1[1]:point2[1],point1[0]:point2[0]] # updated object roi
		object_window=(point1[0],point1[1],w,h) # updated object window
		new_frame1=copy.deepcopy(new_frame) 
		bg_img =mask_bg(object_window,new_frame1) # updated surrounding image
		#cv2.imshow("new_img",new_img)
		#cv2.waitKey(10)
		#cv2.destroyAllWindows()

		# checking the condition whether an object cand may be a distractor
		#####
		distractor_mask=np.where(list_score>lamda_v*score2_objcand) 
		distractor_mask=np.array(distractor_mask)
		distractor_mask[0]=distractor_mask[0]+y0
		distractor_mask[1]=distractor_mask[1]+x0
		dist_img_list=[] # this list will be containing updated distractors
		dist_img_points=[(point1[0],point1[1])]
		for n in range(len(distractor_mask[0])) :
			count_dist=0
			dx=distractor_mask[1][n]
			dy=distractor_mask[0][n]

			for l in range(len(dist_img_points)) :
				diffx=dx-dist_img_points[l][0]
				diffy=dy-dist_img_points[l][1]
				if (diffx>w or diffx<-w or diffy>h or diffy<-h) :
					count_dist=count_dist+1
				else :
					w_box=w-math.sqrt(diffx*diffx)	
					h_box=h-math.sqrt(diffy*diffy)
					area=w_box*h_box
					if(area<0.1*w*h) :
						count_dist=count_dist+1
						
			if(count_dist==len(dist_img_points)) :
				distractor=new_frame[dy:dy+h,dx:dx+w]
				dist_img_list.append(distractor)
				dist_img_points.append((dx,dy))
		##################### distractors updated
		
		hist_obj = cv2.calcHist([obj_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		hist_bg  = cv2.calcHist([bg_img],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
		# removing the effect of pixels having value (0,0,0) from bg which were present in object-region
		hist_bg[0][0][0]=hist_bg[0][0][0]- np.sum(hist_obj) 
		hist_D=np.empty_like(hist_obj)
		hist_D=hist_D.astype('float')
	
		for m in range(len(dist_img_list)) :
			hist_dist=cv2.calcHist([dist_img_list[m]],[0,1,2],None,[bin,bin,bin],[0,256,0,256,0,256])
			hist_D1=hist_D
			check_nan=np.isnan(hist_D)
			hist_D[check_nan]=0
			hist_D=hist_D+hist_dist	

		# probabilities of ob-surr and obj-dist model
		prob_S=prob_obj(hist_obj,hist_bg)
		prob_D=prob_obj(hist_obj,hist_D)

		if (len(dist_img_list)==0) :
			prob_comb_new=prob_S
		else :
			prob_comb_new=prob_D*lamda+prob_S*(1-lamda)	
		prob_comb=update_model(prob_comb,prob_comb_new)

		print "fps",1/(time.time()-t) # computing frames per second
		#computing likelihood maps that will be used for scale estimation
		color_map_surr,Prob_surr=likelihood_map((hist_obj,hist_bg),new_frame) # call likelihood function for obj-surr model
		color_map_dist,Prob_dist=likelihood_map((hist_obj,hist_D),new_frame)# call likelihood function for obj-dist model
		Prob_comb=Prob_surr*(1-lamda)+Prob_dist*lamda
		final_map=Prob_comb*255
		color_map_final_gray=final_map.astype('uint8')
		#using combined likelihood map as a grayscale image to compute adaptive threshold and then getting rois from it 
		obj_img_gray=color_map_final_gray[point1[1]:point2[1],point1[0]:point2[0]]
		gray_copy=copy.deepcopy(color_map_final_gray)
		bg_img_gray =mask_bg(object_window,gray_copy)
		# histograms for object and surrounding regions
		hist_obj_gray=cv2.calcHist([obj_img_gray],[0],None,[256],[0,256]) 
		hist_bg_gray =cv2.calcHist([bg_img_gray],[0],None,[256],[0,256])
		hist_bg_gray[0]=hist_bg_gray[0]-np.sum(hist_obj_gray)
		# normalization
		hist_obj_gray=hist_obj_gray/np.sum(hist_obj_gray)
		hist_bg_gray=hist_bg_gray/np.sum(hist_bg_gray)
		# Cumulative histogram
		cum_bg=np.cumsum(hist_bg_gray)
		cum_obj=np.cumsum(hist_obj_gray)
		# objective function--this need to be minimised to get the adaptive threshold
		objective_fn = 2*cum_obj[:len(cum_obj)-1] - cum_obj[1:] + cum_bg[:len(cum_obj)-1]
		# this condition should hold for the above threshold
		condition_fn = cum_obj[:len(cum_obj)-1] + cum_bg[:len(cum_obj)-1]
		select_mask=np.where(condition_fn>1)
		threshold2=np.argmin(select_mask)
		threshold= select_mask[0][threshold2] # adaptive threshold value
		ret,thresh1 = cv2.threshold(color_map_final_gray,threshold,255,cv2.THRESH_BINARY)

		# performing connected component analysis on the binary image
		connectivity=4
		output = cv2.connectedComponentsWithStats(thresh1, connectivity, cv2.CV_32S)
		num_labels = output[0]
		# The second cell is the label matrix
		labels = output[1]
		# The third cell is the stat matrix
		stats = output[2]
		# The fourth cell is the centroid matrix
		centroids = output[3]
		cx=point1[0]+w/2
		cy=point1[1]+h/2
		diff_centx=centroids[:,0]-cx
		diff_centy=centroids[:,1]-cy
		diffx_y = diff_centy*diff_centy + diff_centx*diff_centx
		diff=np.sqrt(diffx_y) # distance between our object's center and the center of the bounding box of the each component
		boolarr= np.where(diff<math.sqrt((h*h+w*w)/2)) # considering only those components satisfying this condition

		# getting the index of the component with area closest to our object hypothesis' area
		area_diff=[]
		for label_num in boolarr[0] :
			area_diff.append(abs(h*w-stats[label_num][4]))
		if(len(area_diff)>0) :	
			area_diff=np.array(area_diff)
			closest_match_index=np.argmin(area_diff)
			# calculating the ratio of areas, if ratio is greater than a limit then scale will not be changed
			ratio_area=h*w/stats[closest_match_index][4]
			if(ratio_area<2 and ratio_area>0.5) :
				# updating the scale of object-model
				h=h*0.8 + stats[closest_match_index][3]*0.2
				w=w*0.8 + stats[closest_match_index][2]*0.2	
		# updating the window along with constraints
		h=int(h)
		w=int(w)
		#cv2.imwrite("result_trellis/" + str(i)+".jpg",new_img)
		x=int(max(0,cx-(w/2)))
		y=int(max(0,cy-(h/2)))
		x1=int(min(w_img-1,cx+(w/2)))
		y1=int(min(h_img-1,cy+(h/2)))
		object_window=(x,y,w,h)
		cv2.rectangle(new_img,(x,y),(x1,y1),(0, 255, 0), 2)
		cv2.imshow("new_img",new_img)
		cv2.waitKey(10)
		i=i+1
