'''
Main file to run the model
'''
import sys




def run_model(name, img):
    pass





if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <model_name> <img_file>")
        sys.exit(1)
    model_name = sys.argv[1]
    img_file = sys.argv[2]

    run_model(model_name, img_file)
    print(f"Running model {model_name} on image {img_file}")