import cv2
import numpy as np

def enhance_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    equ = cv2.equalizeHist(gray)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # Sharpening kernel
    sharpened = cv2.filter2D(equ, -1, kernel)
    
    # Apply noise reduction (Gaussian blur)
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    
    return blurred

# Load the face image
image_path = 'detected_faces\\face_0_3.jpg'
image = cv2.imread(image_path)

# Enhance the face image
enhanced_image = enhance_face(image)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
