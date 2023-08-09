import sys
import PIL as Image
import io

sys.path.append("..")

import application.Classifier as classifier


if __name__ == "__main__":
    with open("sample.png", "rb") as image:
        f = image.read()
        b = bytearray(f)
    classifier_model = classifier.classifier()
    image_class, image_class_name = classifier.preprocessing_and_predict(classifier_model, b)
    print(image_class_name)