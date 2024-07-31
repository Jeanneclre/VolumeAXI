import torch
from nets.classification import Net, NetTarget, NetFC

# # # Load the entire checkpoint
# # checkpoint = torch.load('Training_Left/SEResNet50/Models/epoch_200-val_loss_0.607.ckpt')

# # # Assuming the class weights were saved under a key named 'class_weights'
# # class_weights = checkpoint['class_weights'] if 'class_weights' in checkpoint else None
# # print("Class Weights:", class_weights)


# # show model architecture

# model = Net(args=None,class_weights=None, base_encoder="SEResNet50", num_classes=3)
# model_path = 'Training_Left/SEResNet50/Models/epoch_200-val_loss_0.607.ckpt'
# model.load_state_dict(torch.load(model_path))
# print('Model Path:', model_path)

import pytorch_lightning as pl

def main(args):
    # Path to your checkpoint
    checkpoint_path = args.model_path

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print('checkpoint state dict:', checkpoint['state_dict'].keys())
    # If you just want to load the model weights into an existing model instance
    model = NetFC()  # Ensure you instantiate your model with the appropriate arguments
    model.load_from_checkpoint(checkpoint_path, strict=False)

    # print('keys:',checkpoint.keys())
    print('model:', model)

    # Check if class weights are stored under a specific key
    if 'class_weights' in checkpoint:
        class_weights = checkpoint['class_weights']
        print("Class Weights:", class_weights)
    else:
        print("Class weights not found in checkpoint.")

    if hasattr(model, 'hparams'):
        print("Hyperparameters:", model.hparams)


# def main(args):
#     #Path to your checkpoint
#     checkpoint_path = args.model_path

#     # Load the checkpoint
#     checkpoint = torch.load(checkpoint_path)

#     print('checkpoint:', checkpoint.keys())
#     print('checkpoint state dict:', checkpoint['state_dict'].keys())
#     model = Net()
#     model.load_state_dict(checkpoint['state_dict'])

#     print('=====MODEL INFO=====')
#     print('Base Encoder:', model.hparams)
#     print('Hyperparameters:', checkpoint['hyper_parameters'])
#     print('====================')
#     print('model:', model)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--nb_class', type=int, default=3)
    args = parser.parse_args()

    main(args)
