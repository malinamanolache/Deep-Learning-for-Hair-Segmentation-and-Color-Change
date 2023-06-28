import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TensorBoard event file
event_file = 'C:\\Users\\Maly\\Desktop\\PyTorch_YOLOv4\\runs\\train\\exp3\\events.out.tfevents.1684787838.Lenovo3.557204.0'
train_loss_obj = []
train_loss_box = []
val_loss_obj = []
map_metric = []

# Extract the loss values from the event file
for event in tf.compat.v1.train.summary_iterator(event_file):
    for value in event.summary.value:
        if value.tag == 'metrics/mAP_0.5':
            map_metric.append((event.step, value.simple_value))
       

# Extract the x and y values for training and validation loss
train_steps, map_m = zip(*map_metric)

# Plot the training loss (object loss and box loss)
plt.figure()
plt.plot(train_steps, map_m)
plt.title('mAP')
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.legend()
plt.savefig('map.png', dpi=400, bbox_inches='tight')
plt.close()

print('Plots saved as train_loss.png and val_loss.png')
