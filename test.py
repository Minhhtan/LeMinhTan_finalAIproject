
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

video=cv2.VideoCapture(0)    
# video=cv2.VideoCapture('VID_20201119_110621.mp4')
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font=cv2.FONT_HERSHEY_COMPLEX
model = load_model('faceMask_model.h5')


predict = [ 'No Mask','Mask']

predict = np.array(predict)

		# {'fm_maskOff': 0, 'fm_maskOn': 1, 'm_maskOff': 2, 'm_maskOn': 3}
count=0	
while True:
	sucess, imgOrignal=video.read()
	imgOrignal = cv2.resize(imgOrignal,[1280,720])
	faces = facedetect.detectMultiScale(imgOrignal,1.9,3)
	for x,y,w,h in faces:
		# cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
		# cv2.rectangle(imgOrignal, (x-50,y-50),(x+w+50, y+h+50), (0,255,255),-2)
		crop_img=imgOrignal[y-80:y+h+50,x-50:x+w+50]
		# crop_img=imgOrignal[y:y+h,x:x+h]
		img = cv2.resize(crop_img, (150, 150))
		img_temp = cv2.resize(crop_img, (150, 150))
		# img=preprocessing(img)
		img = img_to_array(img)
		
		img=img.reshape(1, 150,150, 3)
		img = img.astype('uint32')
		img = img/255
		# cv2.putText(imgOrignal, "Class" , (20,35), font, 0.75, (0,0,255),2, cv2.LINE_AA)
		# cv2.putText(imgOrignal, "Probability" , (20,75), font, 0.75, (255,0,255),2, cv2.LINE_AA)
		prediction=model.predict(img)
		kq = np.argmax(model.predict(img))
		if kq==1:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(predict[kq]),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
		elif kq==0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
			cv2.putText(imgOrignal, str(predict[kq]),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			

	cv2.imshow("Detect FaceMask",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()


















