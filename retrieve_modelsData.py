import torch
from nets.classification import Net

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

# Path to your checkpoint
checkpoint_path = 'Training_Left/SEResNet50/Models/epoch_200-val_loss_0.607.ckpt'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# If you just want to load the model weights into an existing model instance
model = Net()  # Ensure you instantiate your model with the appropriate arguments
model.load_from_checkpoint(checkpoint_path)

print(checkpoint.keys())

# Check if class weights are stored under a specific key
if 'class_weights' in checkpoint:
    class_weights = checkpoint['class_weights']
    print("Class Weights:", class_weights)
else:
    print("Class weights not found in checkpoint.")

if hasattr(model, 'hparams'):
    print("Hyperparameters:", model.hparams)

print('Model data:', model)