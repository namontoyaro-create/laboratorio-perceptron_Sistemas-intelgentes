from src.perceptron import Perceptron
from src.dataset import training_data

model = Perceptron()
model.train(training_data)

prediction = model.predict([1, 0])

print(prediction)