# Cotton Leaf Disease Detection Model using Custom CNN


<p align="center">
    <img src=https://github.com/user-attachments/assets/e598abd7-4037-45b5-b5a3-159c7c304091>
</p>

A deep learning model built with TensorFlow to detect various diseases in cotton leaves using a custom Convolutional Neural Network (CNN) architecture based on LeNet. The model can identify six different conditions: Aphids, Army worm, Bacterial Blight, Healthy leaves, Powdery Mildew, and Target spot.

## Model Performance
- **Accuracy**: 99.43%
- **Loss**: 0.0915

## Features
- Custom CNN architecture based on LeNet
- Batch normalization for better training stability
- Dropout layers for preventing overfitting
- L2 regularization for weight optimization
- Real-time image prediction capabilities
- Support for multiple cotton leaf conditions

## Dataset Structure
The dataset is organized into six classes:
1. Aphids
2. Army worm
3. Bacterial Blight
4. Healthy
5. Powdery Mildew
6. Target spot

## Model Architecture
The model uses a modified LeNet architecture with the following key components:

```python
- Input Layer (256x256x3)
- Convolutional Layer (6 filters, 3x3 kernel)
- Batch Normalization
- MaxPooling Layer (2x2)
- Dropout (0.5)
- Flatten Layer
- Dense Layer (100 units)
- Batch Normalization
- Dropout (0.5)
- Dense Layer (10 units)
- Batch Normalization
- Output Layer (6 units, softmax)
```

## Requirements
- TensorFlow 2.0+
- NumPy
- Matplotlib
- scikit-learn
- PIL (Python Imaging Library)
- IPython

## Usage
1. **Dataset Preparation**
```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Main dataset",
    shuffle=True,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
)
```

2. **Training the Model**
```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    batch_size=32,
    epochs=10
)
```

3. **Making Predictions**
```python
# For single image prediction
predicted_class, confidence = predict_image_with_model(image_data, model)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")
```

## Training Details
- Image Size: 256x256 pixels
- Batch Size: 32
- Training Split: 80%
- Validation Split: 10%
- Test Split: 10%
- Epochs: 10
- Dropout Rate: 0.5
- L2 Regularization Rate: 0.01

## Results
The model achieves high accuracy in detecting various cotton leaf diseases:
- Test Accuracy: 99.43%
- Test Loss: 0.0915

## Future Improvements
1. Implement data augmentation for better generalization
2. Experiment with different CNN architectures
3. Add support for mobile deployment
4. Implement gradual learning rate decay
5. Add model interpretability features

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
