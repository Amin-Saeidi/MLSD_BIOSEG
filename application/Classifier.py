from PIL import Image
from torchvision import models, transforms
import torch
import io
import requests
import os
import sys
sys.path.append("..")


def classifier():
    # Define the ResNet model with the same architecture as your finetuned model
    model = models.resnet50(num_classes=2) # Replace 2 with the number of classes in your finetuned model


    if not os.path.isfile("./var/lib/data/resnet_ft.pt"):

        url = 'https://drive.usercontent.google.com/download?id=19pOxhX9t-1fbJuAF8EHMEnEcdvM2ruAL&export=download&authuser=0&confirm=t&uuid=944c801c-735f-4c8a-9342-48eb8c426395&at=AC2mKKTGKC3HpcBooV6yXcunzL88:1691560364241'
        r = requests.get(url, allow_redirects=True)
        open('./var/lib/data/resnet_ft', 'wb').write(r.content)
        print("RES-net Model Added to Disk")
        logging.info('RES-net Model Added to Disk')


    # Load the saved weights into the model
    checkpoint = torch.load('./var/lib/data/resnet_ft.pt', map_location=torch.device('cpu'))
    # print(checkpoint.keys())  # Print the keys of the checkpoint dictionary
    model.load_state_dict(checkpoint)  # Replace with the actual key name in your checkpoint dictionary
    # print(model)

    return model


def preprocessing_and_predict(model, image_bytes):

    model.eval()
    # print("Before PIL: ", type(image_bytes))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # print("PIL: ", type(image))
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()])

    tensor = transform(image).unsqueeze(0)
    output = model(tensor)

    _, predicted = torch.max(output.data, 1)
    # print(predicted)
    if predicted[0] == 1:
        class_name = "CELL MEMBRANE"
    else:
        class_name = "OTHER"

    return predicted, class_name

# if __name__ == "__main__":
#     classifier()
