import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TensorBoard event file
event_file = 'C:\\Users\\Maly\\Desktop\\PyTorch_YOLOv4\\runs\\train\\exp3\\events.out.tfevents.1684787838.Lenovo3.557204.0'
train_loss_obj = []
train_loss_box = []
val_loss_obj = []
val_loss_box = []

# Extract the loss values from the event file
for event in tf.compat.v1.train.summary_iterator(event_file):
    for value in event.summary.value:
        if value.tag == 'train/obj_loss':
            train_loss_obj.append((event.step, value.simple_value))
        elif value.tag == 'train/box_loss':
            train_loss_box.append((event.step, value.simple_value))
        elif value.tag == 'val/obj_loss':
            val_loss_obj.append((event.step, value.simple_value))
        elif value.tag == 'val/box_loss':
            val_loss_box.append((event.step, value.simple_value))

# Extract the x and y values for training and validation loss
train_steps_obj, train_loss_obj = zip(*train_loss_obj)
train_steps_box, train_loss_box = zip(*train_loss_box)
val_steps_obj, val_loss_obj = zip(*val_loss_obj)
val_steps_box, val_loss_box = zip(*val_loss_box)

# Plot the training loss (object loss and box loss)
plt.figure()
plt.plot(train_steps_obj, train_loss_obj, label='Train Object Loss')
plt.plot(train_steps_box, train_loss_box, label='Train Box Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_loss.png', dpi=400, bbox_inches='tight')
plt.close()

# Plot the validation loss (object loss and box loss)
plt.figure()
plt.plot(val_steps_obj, val_loss_obj, label='Validation Object Loss')
plt.plot(val_steps_box, val_loss_box, label='Validation Box Loss')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('val_loss.png', dpi=400, bbox_inches='tight')
plt.close()

print('Plots saved as train_loss.png and val_loss.png')
