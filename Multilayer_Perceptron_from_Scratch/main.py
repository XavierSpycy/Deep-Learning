import matplotlib.pyplot as plt
from MLPClassifier import Dataset, HiddenLayer, MLP

X_train, y_train, X_valid, y_valid, X_test, y_test = Dataset(random_state=0).load(normalize=True)

layer1 = HiddenLayer(128, 48, activation='softplus')
layer2 = HiddenLayer(48, 48, batch_norm=True)
layer3 = HiddenLayer(48, 24, activation='softplus')
layer4 = HiddenLayer(24, 24, batch_norm=True)
layer5 = HiddenLayer(24, 10, activation='softmax')
layers = [layer1, layer2, layer3, layer4, layer5]
nn = MLP(layers, optimizer='adagrad', hyperparams={'learning_rate': 0.001})
loss = nn.fit(X_train, y_train, epochs=500, batch_size=32)

print("Accuracy on training set: {:.2f}%".format(nn.accuracy_score(X_train, y_train)*100))
print("Accuracy on validation set: {:.2f}%".format(nn.accuracy_score(X_valid, y_valid)*100))
print("Training time: {:.0f} sec".format(nn.train_time))
print("Loss: {:.4f}".format(loss[-1]))
plt.plot(loss)