from helpers.process_image import process_image
from helpers.load_checkpoint import load_checkpoint
import argparse
import torch
import torch.nn as nn
import numpy as np
import json


def predict(image_path, checkpoint, topk=5, category_names=None, cuda=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model = load_checkpoint(checkpoint, cuda=cuda)
    # Use GPU if it's available
    device = torch.device("cuda" if cuda else "cpu")
    model.to(device)
    
    # Set model to evaluation mode
    model_p = model.eval()
    
    # Process image
    img = process_image(image_path)
    img = img.numpy()
    # convert image to a tensor with size [1, 3, 224, 224]
    img = torch.from_numpy(np.array([img])).float()
    
    with torch.no_grad():
        output = model_p.forward(img.to(device))
    
    ps = torch.exp(output).data
    
    top_p, top_class = ps.topk(topk, dim=1)
    idx_to_class = model.idx_to_class

    # Get classes names if exists
    class_names = "Not defined"
    if category_names is not None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        class_names = [cat_to_name[idx_to_class[idx]] for idx in np.array(top_class[0])]
    

    return top_p, top_class, class_names

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability')
    parser.add_argument('image_path', type=str, help='Image path')
    parser.add_argument('checkpoint', type=str, help='Models checkpoint for inference')
    parser.add_argument('--top_k', type=int, help='Return top k most likely classes')
    parser.add_argument('--category_names', type=str, help='json file giving a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

    # Parse and read arguments and assign them to variables if exists 
    args, _ = parser.parse_known_args()

    image_path = args.image_path
    checkpoint = args.checkpoint

    top_k = 5
    if args.top_k:
        top_k = args.top_k

    category_names = None
    if args.category_names:
        category_names = args.category_names

    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("Warning! GPU flag was set however GPU is not available")

    probs, classes, class_names = predict(image_path, checkpoint, topk=top_k, category_names=category_names, cuda=cuda)
    print("="*80)
    print(" "*35 + 'FLOWER PREDICTOR')
    print("="*80)
    print("Class(es) name(s) = {}".format(class_names))
    print("Probability confidence(s) = {}".format(probs[0]))
    print("="*80)
    

if __name__ == '__main__':
    main()