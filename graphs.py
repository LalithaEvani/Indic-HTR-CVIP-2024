from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# Load the TensorBoard log file
log_file = 'outputs/parseq/2024-05-21_16-58-44'  # Update with the path to your TensorBoard log directory
event_acc = EventAccumulator(log_file)
event_acc.Reload()

epochs_steps = []
epochs_value = []

for epoch in event_acc.Scalars('epoch'):
    epochs_steps.append(epoch.step)
    epochs_value.append(epoch.value)

event_acc.Reload()

loss_steps = []
loss_value = []

for loss in event_acc.Scalars('loss'):
    loss_steps.append(loss.step)
    loss_value.append(loss.value)

event_acc.Reload()

val_loss_steps = []
val_loss_value = []

for val_loss in event_acc.Scalars('val_loss'):
    val_loss_steps.append(val_loss.step)
    val_loss_value.append(val_loss.value)

event_acc.Reload()



plt.figure(figsize=(8, 6))
plt.plot(loss_steps, loss_value, label='train_loss')
plt.plot(val_loss_steps,val_loss_value, label='val_loss')
plt.title('epochs vs iterations')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.savefig('outputs/parseq/bengali_graphs.png')


