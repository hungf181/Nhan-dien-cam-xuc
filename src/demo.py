#from keras.models import load_model
from tensorflow import keras
model = keras.models.load_model("Emotion-detection/src/facenet_keras.h5")
print('Done!')