'''
Main file to run the model
'''
import sys
import torch
from torchvision import transforms
from PIL import Image
from models.CNN import CNNModel
import torch.nn.functional as F



def run_model(name, img):
    '''
    Remember to add the other architectures
    '''

    targets = ['Dirt 💩', "Grass 🌱", "Rock 🪨", "Sand 🏖️"]

    if name == 'model-cnn.pth':
        model = CNNModel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(name, map_location=torch.device('cpu'), weights_only=True))
    model.eval()  

    preprocess = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(img)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  

    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)
        i = torch.argmax(probabilities, dim=1).item()
        print()
        print('----------------------------------------------------------------------------')
        print(f"Predicted class: {targets[i]}")
        print('----------------------------------------------------------------------------')



    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <model_name> <img_file>")
        sys.exit(1)
    model_name = sys.argv[1]
    img_file = sys.argv[2]

    print(f"Running model {model_name} on image {img_file}")
    run_model(model_name, img_file)
    