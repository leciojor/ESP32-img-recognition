'''
Main file to run the model
'''
import sys
import torch
from torchvision import transforms
from PIL import Image
from models.CNN import CNNModel
from models.Layered import finalConnectedNetwork, weighted_voting
import numpy as np
import torch.nn.functional as F


def get_prediction(model, name, img, targets, v=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load("models/" + name, map_location=device, weights_only=True))
    model.eval()  

    preprocess = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
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
        
        if v:
            print()
            print('----------------------------------------------------------------------------')
            print(f"Predicted class: {targets[i]}")
            print('----------------------------------------------------------------------------')
        return probabilities


def run_model(names, img, k=3):
    '''
    Remember to add the other architectures
    '''

    targets = ['Dirt 💩', "Grass 🌱", "Rock 🪨", "Sand 🏖️"]
    if len(names) == 2:
        model1 = CNNModel()
        model2 = finalConnectedNetwork(k=k)
        name1 = names[0]
        name2 = names[1]
        prob1 = get_prediction(model1, name1, img, targets, v=False).tolist()
        prob2 = get_prediction(model2, name2, img, targets, v=False).tolist()
        print(prob1)
        i = weighted_voting(np.array([prob1, prob2]), [0.7, 0.3])[0]
        print()
        print('----------------------------------------------------------------------------')
        print(f"Predicted class: {targets[i]}")
        print('----------------------------------------------------------------------------')


    else:
        name = names[0]
        if 'model-cnn' in name:
            model = CNNModel()
        if 'layered' in name: 
            model = finalConnectedNetwork(k=k)
        get_prediction(model, name, img, targets)
        
    



if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python run.py <model1_name> <model2_name> <img_file>")
        sys.exit(1)

    models = []
    if len(sys.argv) == 3:
        name = sys.argv[1]
        models.append(name)
        img_file = sys.argv[2]
        print(f"Running model {name} on image {img_file}")
    else:
        models.append(sys.argv[1])
        models.append(sys.argv[2])
        img_file = sys.argv[3]
        print(f"Running models on image {img_file} with ensemble mode")

    
    run_model(models, img_file)
    