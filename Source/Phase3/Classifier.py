from PIL import Image
from torchvision import models, transforms
import torch
import io

def classifier():
    # Define the ResNet model with the same architecture as your finetuned model
    model = models.resnet50(num_classes=2) # Replace 2 with the number of classes in your finetuned model

    # Load the saved weights into the model
    checkpoint = torch.load('resnet_ft.pt', map_location=torch.device('cpu'))
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
    print(predicted)
    if predicted[0] == 1:
        class_name = "CELL MEMBRANE"
    else:
        class_name = "OTHER"

    return predicted, class_name

# if __name__ == "__main__":
#     classifier()