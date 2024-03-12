import cv2
import tensorflow
import numpy

vid=cv2.videoCapture(0)
model = tensorflow.keras.models.load_model("keras_model.h5")
while(True):
    ret, frame = vid.read()
    imagestorage = cv2.resize(frame,(224,224))
    storeimg = numpy.array(imagestorage, dtype=numpy.float32)
    storeimg = numpy.expand_dims(storeimg,axis=0)
    normal= storeimg/255.0
    predict = model.predict(normal)
    print(predict)

    cv2.imshow('frame', frame)
      
  
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  

vid.release()

cv2.destroyAllWindows()
