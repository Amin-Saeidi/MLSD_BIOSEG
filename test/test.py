import sys
import logging

sys.path.append("..")

import application.Classifier as classifier

if __name__ == "__main__":
    
    with open("sample.png", "rb") as image:
        f = image.read()
        b = bytearray(f)
    
    classifier_model = classifier.classifier("test")
    image_class, image_class_name = classifier.preprocessing_and_predict(classifier_model, b)
    
    if image_class_name == "CELL MEMBRANE":
        test_checker = True
    else:
        test_checker = False

    if test_checker == False:
        raise Exception("Sorry, Model is not performing well! - please check the classifier model")
    else:
        print("MODEL IS ALRIGHT! - ALL THE TESTS PASSED :)")
