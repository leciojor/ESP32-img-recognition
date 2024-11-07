from PIL import Image, ImageEnhance, ImageStat
import cv2
import numpy as np
import json
from tqdm import tqdm
import torchvision.transforms as transforms



def is_too_bright(img, thr = 220):
    grayscale_image = img.convert("L")
    stat = ImageStat.Stat(grayscale_image)
    avg = stat.mean[0]
    return avg > thr


def removing_too_bright(img, b=190):
    if is_too_bright(img):
        brightness_enhancer = ImageEnhance.Brightness(img)
        factor = b / ImageStat.Stat(img.convert("L")).mean[0]
        adj = brightness_enhancer.enhance(factor)
        return adj
    
def img_tensor(img):
    tensor = np.array(img)

    return tensor


def removing_salt_pepper(img_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filtered = cv2.medianBlur(img, ksize=3)  
    fil = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(fil)

    return img


def augment(img, n = 60):

    finals = []
    for _ in tqdm(range(n), desc="Rotating"):
        transform = transforms.Compose([
        transforms.RandomChoice([
        transforms.RandomRotation(degrees=15, fill=(255, 255, 255)), 
        transforms.RandomHorizontalFlip(),                           
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)])])  

        transformed = transform(img)
        finals.append(transformed)

        

    return finals


def prepare(img):
    #resizing
    img = img.resize((640,480), Image.Resampling.LANCZOS)

    return img

def cleaning(img, img_file):
    img = removing_too_bright(img)
    img = removing_salt_pepper(img_file)

    return img


def main():
    raw1 = 'data/raw/train/'
    raw2 = 'data/raw/test/'
    ref1 = []
    ref2 = []

    with open(raw1 + "metadata.jsonl", "r") as file:
        for line in file:
            entry = json.loads(line.strip())
            ref1.append(entry)
    
    with open(raw2 + "metadata.jsonl", "r") as file:
        for line in file:
            entry = json.loads(line.strip())
            ref2.append(entry)
    count = 0

    print(ref1)
    print(ref2)

    for img_file in tqdm(ref1, desc = "processing data 1"):
        file_path = raw1 + img_file['file_name'][:-3] +'png'
        print(file_path)
        image = Image.open(file_path)
        img = prepare(image)
        img = cleaning(img, file_path)
        finals = augment(img)

        for final in finals:
            count += 1
            if "set01" in img_file['file_name'] or "set03" in img_file['file_name'] or "set05" in img_file['file_name']:
                final.save(f"data/full_clean/rocky/img{count}.png")
            elif "set02" in img_file['file_name'] or "set04" in img_file['file_name'] or "set06" in img_file['file_name'] or "set09" in img_file['file_name'] or "set17" in img_file['file_name']:
                final.save(f"data/full_clean/grassy/img{count}.png")
            elif "set11" in img_file['file_name'] or "set14" in img_file['file_name'] or "set16" in img_file['file_name']:
                final.save(f"data/full_clean/sandy/img{count}.png")
            elif "set08" in img_file['file_name'] or "set10" in img_file['file_name'] or "set12" in img_file['file_name'] or "set13" in img_file['file_name'] or "set15" in img_file['file_name']:
                final.save(f"data/full_clean/dirt/img{count}.png")

    for img_file in tqdm(ref2, desc = "processing data 2"):
        file_path = raw2 + img_file['file_name'][:-3] +'png'
        print(file_path)
        image = Image.open(file_path)
        img = prepare(image)
        img = cleaning(img, file_path)
        finals = augment(img)

        for final in finals:
            count += 1
            if "set01" in img_file['file_name'] or "set03" in img_file['file_name'] or "set05" in img_file['file_name']:
                final.save(f"data/full_clean/rocky/img{count}.png")
            elif "set02" in img_file['file_name'] or "set04" in img_file['file_name'] or "set06" in img_file['file_name'] or "set09" in img_file['file_name'] or "set17" in img_file['file_name']:
                final.save(f"data/full_clean/grassy/img{count}.png")
            elif "set11" in img_file['file_name'] or "set14" in img_file['file_name'] or "set16" in img_file['file_name']:
                final.save(f"data/full_clean/sandy/img{count}.png")
            elif "set08" in img_file['file_name'] or "set10" in img_file['file_name'] or "set12" in img_file['file_name'] or "set13" in img_file['file_name'] or "set15" in img_file['file_name']:
                final.save(f"data/full_clean/dirt/img{count}.png")





if __name__ == '__main__':
    main()