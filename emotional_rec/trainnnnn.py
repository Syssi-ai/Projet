import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
#vGG16 cest le modele pre entrainee (comme resnet18) 
#cifar, imagenet sont des dataset 
#vgg16 et resnet18 = modeles pre-entraines


model = VGG16(weights='imagenet')
#on a declare une variable model, dans laquelle on a importe le modele VGG16
#weights = poids 
#weights cest le truc quapprend le modele 
img_path = "Happy.jpg"  
img = Image.open(img_path).convert('RGB')
img = img.resize((224, 224)) #224x224 pixels = taille de limage

img_array = np.array(img) #create tableau 4 images
img_array = np.expand_dims(img_array, axis=0) 
img_array = preprocess_input(img_array)

preds = model.predict(img_array) #on a declare une variable preds
#dans laquelle on a sauvgarde le resultat predit par le modele
decoded = decode_predictions(preds, top=5)[0]

#on prend le top5 des meilleures classes predites


plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('on')

#afficher l'image avec la fonction plt
title = "Predictions:\n"
for label, name, prob in decoded:
    title += f"{name}: {prob*100:.1f}%\n"

plt.title(title)
plt.show()
