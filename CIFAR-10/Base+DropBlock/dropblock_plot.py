import json
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


NAMES = ["A_1_X",
             "A_1_Y",
             "A_1_W",
             "A_1_Z",
             "B_1_X",
             "B_1_Y",
             "B_1_W",
             "B_1_Z",
             "C_1_X",
             "C_1_Y",
             "C_1_W",
             "C_1_Z",
             "D_1_X",
             "D_1_Y",
             "D_1_W",
             "D_1_Z",
             "A_2_X",
             "A_2_Y",
             "A_2_W",
             "A_2_Z",
             "B_2_X",
             "B_2_Y",
             "B_2_W",
             "B_2_Z",
             "C_2_X",
             "C_2_Y",
             "C_2_W",
             "C_2_Z",
             "D_2_X",
             "D_2_Y",
             "D_2_W",
             "D_2_Z",
             "A_3_X",
             "A_3_Y",
             "A_3_W",
             "A_3_Z",
             "B_3_X",
             "B_3_Y",
             "B_3_W",
             "B_3_Z",
             "C_3_X",
             "C_3_Y",
             "C_3_W",
             "C_3_Z",
             "D_3_X",
             "D_3_Y",
             "D_3_W",
             "D_3_Z",
             "A_4_X",
             "A_4_Y",
             "A_4_W",
             "B_4_X",
             "B_4_Y",
             "B_4_W",
             "C_4_X",
             "C_4_Y",
             "C_4_W",
             "D_4_X",
             "D_4_Y",
             "D_4_W",
             "A_5_X",
             "A_5_Y",
             "A_5_W",
             "B_5_X",
             "B_5_Y",
             "B_5_W",
             "C_5_X",
             "C_5_Y",
             "C_5_W",
             "D_5_X",
             "D_5_Y",
             "D_5_W"]

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
       epoch_at_best_val_accuracy = -100000
       best_val_accuracy = max(history["val_accuracy"][:20])
       epoch_at_best_val_accuracy = history["val_accuracy"][:20].index(best_val_accuracy)
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

print("BESTS")
for best_val_accuracy in best_val_acc:
    print(best_val_accuracy)

print(best_val_acc)

best_idx = best_val_acc.index(max(best_val_acc))
best_history = histories[best_idx]
print(NAMES[best_idx])
print(best_val_acc[best_idx])
'''
x = [1, 2, 3, 4, 5]
y = [0.01, 0.1, 0.2, 0.5]

validation_accuracies = format_heatmap_data(best_val_acc)

ax = sns.heatmap(validation_accuracies, yticklabels = ["0.5", "0.2", "0.1", "0.01"], xticklabels = ["1", "2", "3", "4", "5"], annot=True, linewidth=0.5, fmt=".3f")
plt.show()

epochs = format_heatmap_data(best_epoch)

ax = sns.heatmap(epochs, yticklabels = ["0.5", "0.2", "0.1", "0.01"], xticklabels = ["1", "2", "3", "4", "5"], annot=True, linewidth=0.5)
plt.show()
print(best_val_acc[11]) #12th one, D3, 0.5 at location 3

best_history = histories[11]
plt.plot(best_history['accuracy'], label='accuracy')
plt.plot(best_history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

best_history = histories[11]
plt.plot(best_history['accuracy'], label='accuracy')
plt.plot(best_history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
'''

base_train_accuracy = np.load("./train_accuracy.npy")
base_train_loss = np.load("./train_loss.npy")
base_val_accuracy = np.load("./val_accuracy.npy")
base_val_loss = np.load("./val_accuracy.npy")

best_history = histories[best_idx]
print("BEST INDEX")
print(best_idx)
print(best_history['val_accuracy'][:20])
print("THIS IS THE BEST")
print(max(best_history['val_accuracy'][:20]))
'''
plt.plot(best_history['accuracy'][:20], label='BlockDrop Model Accuracy')
plt.plot(best_history['val_accuracy'][:20], label = 'BlockDrop Model Validation Accuracy')
plt.plot(base_train_accuracy, label='Base Model Accuracy')
plt.plot(base_val_accuracy, label = 'Base Model Validation Accuracy')
plt.title("Accuracy Comparisons Between the Base Model and Best DropBlock Model")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
#plt.show()

best_history = histories[11]
plt.plot(best_history['loss'][:20], label='DropBlock Model Loss')
plt.plot(best_history['val_loss'][:20], label = 'DropBlock Model Validation Loss')
plt.plot(base_train_loss, label='Base Model Loss')
plt.plot(base_val_loss, label = 'Base Model Validation Loss')
plt.title("Loss Comparisons Between the Base Model and Best DropBlock Model")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
#plt.show()
'''


layer_1 = np.average(best_val_acc[0:16])
layer_2 = np.average(best_val_acc[16:32])
layer_3 = np.average(best_val_acc[32:48])
layer_4 = np.average(best_val_acc[48:60])
layer_5 = np.average(best_val_acc[60:])
print("HEEEEEEEEEEEEERE")
layer_average = [layer_1, layer_2, layer_3, layer_4, layer_5]
print(layer_average)
plt.bar(['1', '2', '3', '4', '5'], layer_average)
#plt.plot(layer_average)
plt.ylim([0.5, 0.8])
plt.title("Average Best Validation Accuracy by Location", fontsize=50)
plt.ylabel("Validation Accuracy", fontsize=40)
plt.xlabel("Prior Convolution Layer Number", fontsize=40)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

print("AAAAAAAAAAAAAAAAAAAAH")
print(best_val_acc)
print(len(best_val_acc))
best_val_acc = np.array(best_val_acc)
rate_1 = np.average(best_val_acc[np.array([0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 60, 61, 62])])
rate_2 = np.average(best_val_acc[np.array([4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 51, 52, 53, 63, 64, 65])])
rate_3 = np.average(best_val_acc[np.array([8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 54, 55, 56, 66, 67, 68])])
rate_4 = np.average(best_val_acc[np.array([12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 57, 58, 59, 69, 70, 71])])
print("HEEEEEEEEEEEEERE")
layer_average = [rate_1, rate_2, rate_3, rate_4]
plt.bar(['0.01', '0.1', '0.2', '0.5'], layer_average)
plt.ylim([0.7, 0.75])
plt.title("Average Best Validation Accuracy by Gamma", fontsize=50)
plt.ylabel("Validation Accuracy", fontsize=40)
plt.xlabel("Gamma Value", fontsize=40)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()








best_val_acc = np.array(best_val_acc)
rate_1 = np.average(best_val_acc[np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 51, 54, 57, 60, 63, 66, 69])])
rate_2 = np.average(best_val_acc[np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 52, 55, 58, 61, 64, 67, 70])])
rate_3 = np.average(best_val_acc[np.array([2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 53, 56, 59, 62, 65, 68, 71])])
rate_4 = np.average(best_val_acc[np.array([3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47])])
print("HEEEEEEEEEEEEERE")
layer_average = [rate_1, rate_2, rate_3, rate_4]
print(layer_average)
plt.bar(['3', '5', '7', '9'], layer_average)
plt.ylim([0.6, 0.78])
plt.title("Average Best Validation Accuracy by BlockSize", fontsize=50)
plt.ylabel("Validation Accuracy", fontsize=40)
plt.xlabel("Block Size", fontsize=40)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
