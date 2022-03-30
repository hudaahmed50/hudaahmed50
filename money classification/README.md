## **Image Detection Vs Image Classification Vs Image Recognition**

As a beginner, it is essential to understand the underlying techniques of computer vision technology. Here’s the breakdown-

a) **Image Detection** is the first step wherein machines detect a certain object in an image. A step further, multiple object detection involves locating several objects in an image by drawing bounding boxes around them.

b) **Image Classification** annotates the detected object with a class label or a category, for example, cat, dog, etc.

c) **Image Recognition** envelopes the above two techniques, training machines to detect, classify, and identify the objects by matching them with given data. For instance, face recognition functionality in smartphones that authenticate a human face by matching it with database input.

## **Inside a Deep Learning Model for Image Recognition**

Under the hood, deep learning models constitute several neural network architectures whose working simulate a human brain’s visual cortex. Specifically, **Region-based Convolutional Neural Networks or R-CNNs** are a family of deep neural networks applied for object localization and image recognition. An R-CNN model constitutes three major modules namely-

**a) Region proposal** for generating candidate bounding boxes

**b) Feature extractor** for extracting features from the identified objects

**c) Classifier** for annotating the object with labels or categories.

**The better the quality of training data, the more accurate and efficient the image recognition model is.**

The most important parameters while training a neural network model include-

**a) Size, quality, and quantity of images**

**b) Number of color channels**

**c) Aspect ratio and image scaling**

**d) The mean and standard deviation of input data**

**e) Available data variations, and more.**

**The better the quality of training data, the more accurate and efficient the image recognition model is.**

The most important parameters while training a neural network model include-

**a) Size, quality, and quantity of images**

**b) Number of color channels**

**c) Aspect ratio and image scaling**

**d) The mean and standard deviation of input data**

**e) Available data variations, and more.**

## **Tech Stack Used for Deep Learning-based Image Recognition**

For enterprises to successfully deploy deep learning for image recognition applications, they must employ effective tools and ML libraries. Below are the most robust toolkit essentials to build image recognition applications-

### **1) TensorFlow**

An open-source machine learning library, TensorFlow has become a star resource for **compiling and executing complex machine learning models**. The comprehensive framework is used for various applications like **image classification and recognition, natural language processing (NLP), and document data extraction**. It can be easily paired with other machine learning tools such as OpenCV to add more value to any machine learning project.

### **2) TensorFlow Lite**

A lighter version of TensorFlow, TensorFlow Lite (.TFLITE) is customarily designed to **run machine learning applications on mobile and edge devices**. With limited memory requirements, TensorFlow Lite disrupts computing constraints and encourages serverless ML development.

However, the framework only facilitates running and not the development of ML models from scratch. The tool is used to convert pre-built and pre-trained ML models on mobile devices.

### **3) Keras**

Keras is a budding neural network library with the ability to run on top of TensorFlow and other ML libraries. Simply put, it is a high-level API capable of deploying TensorFlow functions parallelly. For deep learning, Keras ensures a convenient and speedy prototyping facility while simplifying complex TensorFlow functions for ML beginners.

---

*With a working knowledge of TensorFlow and Keras, **the Oodles AI team** can efficiently deploy these ML frameworks for various enterprise applications. The next section elaborates on such dynamic applications of deep learning for image recognition.*

# **Importing Libraries**

In [1]:

# 

`import osimport globimport numpy as npimport pandas as pdimport seaborn as snsimport matplotlib.pyplot as pltfrom sklearn.model_selection import train_test_splitfrom tensorflow.keras.preprocessing.image import ImageDataGeneratorfrom tensorflow.keras.layers import Densefrom tensorflow.keras.models import Modelfrom tensorflow.keras.callbacks import Callback, EarlyStoppingfrom sklearn.metrics import confusion_matrix, classification_report`

****Load Dataset****

```
labels =list(map(lambdax:os.path.split(os.path.split(x)[0])[1],path_img))
file_path =pd.Series(path_img, name='File_Path').astype(str)
labels =pd.Series(labels, name='Labels')
data =pd.concat([file_path,labels], axis=1)
data =data.sample(frac=1).reset_index(drop=True)
data.head()
```

# **EDA & Visualization**

In [4]:

# linkcode

`fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8),
                        subplot_kw={'xticks': [], 'yticks': []})
for i, ax **in** enumerate(axes.flat):
    ax.imshow(plt.imread(data.File_Path[i]))
    ax.set_title(data.Labels[i])
plt.tight_layout()
plt.show()`

```
counts =data.Labels.value_counts()
sns.barplot(x=counts.index, y=counts)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=15);
```

****Function****

```
deffunc(pre,name_model,size):
train_datagen =ImageDataGenerator(preprocessing_function=pre, validation_split=0.2)
test_datagen =ImageDataGenerator(preprocessing_function=pre)

train_gen =train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='File_Path',
        y_col='Labels',
        target_size=(size,size),
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
valid_gen =train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='File_Path',
        y_col='Labels',
        target_size=(size,size),
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0,
        subset='validation',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
test_gen =test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='File_Path',
        y_col='Labels',
        target_size=(size,size),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        verbose=0,
        shuffle=False)

pre_model =name_model(input_shape=(size,size, 3),
                   include_top=False,
                   weights='imagenet',
                   pooling='avg')
pre_model.trainable =Falseinputs =pre_model.inputx =Dense(64, activation='relu')(pre_model.output)
x =Dense(64, activation='relu')(x)
outputs =Dense(11, activation='softmax')(x)
model =Model(inputs=inputs, outputs=outputs)
model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
my_callbacks  = [EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              mode='auto')]

history =model.fit(train_gen,validation_data=valid_gen,epochs=100,callbacks=my_callbacks,verbose=0);
print('\033[01m              Plotting Accuracy, val_accuracy, loss, val_loss\033[0m')
# Plotting Accuracy, val_accuracy, loss, val_lossfig,ax =plt.subplots(1, 2, figsize=(10, 3))
ax =ax.ravel()

    fori,metinenumerate(['accuracy', 'loss']):
ax[i].plot(history.history[met])
ax[i].plot(history.history['val_' +met])
ax[i].set_title('Model{}'.format(met))
ax[i].set_xlabel('epochs')
ax[i].set_ylabel(met)
ax[i].legend(['Train', 'Validation'])
plt.show()

# Predict Data Testpred =model.predict(test_gen )
pred =np.argmax(pred,axis=1)
labels = (train_gen.class_indices)
labels =dict((v,k) fork,vinlabels.items())
pred = [labels[k] forkinpred]

print('\033[01m              Classification_report\033[0m')
cm=confusion_matrix(test_df.Labels,pred)
clr =classification_report(test_df.Labels,pred)
print(clr)
print('\033[01m Display 8 pictures of the dataset with their labels\033[0m')
fig,axes =plt.subplots(nrows=2, ncols=4, figsize=(12, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

    fori,axinenumerate(axes.flat):
ax.imshow(plt.imread(test_df.File_Path.iloc[i+1]))
        if (test_df.Labels.iloc[i+1]==pred[i+1]):
ax.set_title(f"True:{test_df.Labels.iloc[i+1]}\nPredicted:{pred[i+1]}",color ="green",fontweight ="bold")
        else:
ax.set_title(f"True:{test_df.Labels.iloc[i+1]}\nPredicted:{pred[i+1]}",color ="red",fontweight ="bold")#\033[31mplt.tight_layout()
plt.show()

print('\033[01m              Results\033[0m')
results =model.evaluate(test_gen, verbose=0)
print("    Test Loss:\033[31m\033[01m{:.5f}\033[30m\033[0m".format(results[0]))
print("Test Accuracy:\033[32m\033[01m{:.2f}%\033[30m\033[0m".format(results[1] * 100))

    returnresults
```



