import json
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
sns.set(font_scale=3)
NAMES = ["A_1",
         "B_1",
         "C_1",
         "D_1",
         "A_2",
         "B_2",
         "C_2",
         "D_2",
         "A_3",
         "B_3",
         "C_3",
         "D_3",
         "A_4",
         "B_4",
         "C_4",
         "D_4",
         "A_5",
         "B_5",
         "C_5",
         "D_5"]

def call_on_model(model, names, fcn):
    return fcn(model)

def call_on_models(models, names, fcn):
    output = []
    for i in range(len(models)):
        output.append(fcn(models[i], names[i]))

    return output

RATES = ['rate = 0.01', 'rate = 0.1', 'rate = 0.2', 'rate = 0.5']
LOCATIONS = ['Location 1', 'Location 2', 'Location 3', 'Location 4', 'Location 5']

LOCATION_TITLES = ["Validation Losses for Various Dropout Rates at Location 1",
                   "Validation Losses for Various Dropout Rates at Location 2",
                   "Validation Losses for Various Dropout Rates at Location 3",
                   "Validation Losses for Various Dropout Rates at Location 4",
                   "Validation Losses for Various Dropout Rates at Location 5"]

RATE_TITLES = ["Validation Losses at Various Locations with Droroput Rate 0.01",
               "Validation Losses at Various Locations with Droroput Rate 0.1",
               "Validation Losses at Various Locations with Droroput Rate 0.2",
               "Validation Losses at Various Locations with Droroput Rate 0.5"]

def make_rate_comparison_plots(histories, plotting_fcn):
    num_rates = len(RATES) # 4
    num_locations = len(LOCATIONS) # 5
    for loc in range(num_locations):
        histories_at_loc = []
        for rate in range(rates):
            histories_at_loc.append(histories[loc*(num_locs-1) + rate])
        plotting_fcn(histories, LOCATION_TITLES[loc])

def make_location_comparison_plots(histories, plotting_fcn):
    num_rates = len(RATES) # 4
    num_locations = len(LOCATIONS) # 5
    for rate in range(rates):
        histories_at_rate = []
        for loc in range(num_locations):
            histories_at_rate.append(histories[rate*(num_rates-1) + loc])
        plotting_fcn(histories, RATE_TITLES[rate])


def get_best_models(histories): 
    best_validation_accuracies = []
    epoch_at_best_validation_accuracies = []
    for history in histories:
       best_val_accuracy = max(history["val_accuracy"][:20])
       epoch_at_best_val_accuracy = history["val_accuracy"][:20].index(best_val_accuracy)
       best_validation_accuracies.append(best_val_accuracy)
       epoch_at_best_validation_accuracies.append(epoch_at_best_val_accuracy)

    return epoch_at_best_validation_accuracies, best_validation_accuracies
'''
def get_best_models(histories): 
    best_validation_accuracies = []
    epoch_at_best_validation_accuracies = []
    for history in histories:
       best_val_accuracy = -100000 
       epoch_at_best_val_accuracy = -100000
       for idx, accuracy in enumerate(history["val_accuracy"][:20]):
           if accuracy > best_val_accuracy:
               best_val_accuracy = accuracy
               epoch_at_best_val_accuracy = idx+1
       best_validation_accuracies.append(best_val_accuracy)
       epoch_at_best_validation_accuracies.append(epoch_at_best_val_accuracy)

    return epoch_at_best_validation_accuracies, best_validation_accuracies
'''

def get_histories():
    histories = []
    for name in NAMES:
        f = open(f"history_{name}.json")
        histories.append(json.load(f))
        f.close()
    return histories

def format_heatmap_data(data):
    return np.flip(np.transpose(np.reshape(np.array(data), (5, 4))), 0)

histories = get_histories()
best_epoch, best_val_acc = get_best_models(histories)
best_idx = best_val_acc.index(max(best_val_acc))
best_history = histories[best_idx]
print("HEEEEEEEEEEEEEEEEEERE")
print(NAMES[best_idx])
print(best_val_acc[best_idx])

x = [1, 2, 3, 4, 5]
y = [0.01, 0.1, 0.2, 0.5]

validation_accuracies = format_heatmap_data(best_val_acc)

ax = sns.heatmap(validation_accuracies, yticklabels = ["0.5", "0.2", "0.1", "0.01"], xticklabels = ["1", "2", "3", "4", "5"], annot=True, linewidth=0.5, fmt=".3f")
ax.tick_params(labelsize=30)
plt.title("Best Validation Accuracies for All Networks", fontsize=50)
plt.xlabel("Prior Max Pooling Layer Number", fontsize=40)
plt.ylabel("Dropout Rate", fontsize=40)
plt.show()


epochs = format_heatmap_data(best_epoch)

ax = sns.heatmap(epochs, yticklabels = ["0.5", "0.2", "0.1", "0.01"], xticklabels = ["1", "2", "3", "4", "5"], annot=True, linewidth=0.5)
plt.show()

print("THIS IS THE BEST")
print(best_val_acc[7]) #12th one, D3, 0.5 at location 3


base_train_accuracy = np.load("./train_accuracy.npy")
base_train_loss = np.load("./train_loss.npy")
base_val_accuracy = np.load("./val_accuracy.npy")
base_val_loss = np.load("./val_accuracy.npy")

best_history = histories[7]
plt.plot(best_history['accuracy'][:20], label='Dropout Model Accuracy')
plt.plot(best_history['val_accuracy'][:20], label = 'Dropout Model Validation Accuracy')
plt.plot(base_train_accuracy, label='Base Model Accuracy')
plt.plot(base_val_accuracy, label = 'Base Model Validation Accuracy')
plt.title("Accuracy Comparisons Between the Base Model and Best Dropout Model")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

best_history = histories[7]
plt.plot(best_history['loss'][:20], label='Dropout Model Loss')
plt.plot(best_history['val_loss'][:20], label = 'Dropout Model Validation Loss')
plt.plot(base_train_loss, label='Base Model Loss')
plt.plot(base_val_loss, label = 'Base Model Validation Loss')
plt.title("Loss Comparisons Between the Base Model and Best Dropout Model")
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

print(base_train_accuracy)
