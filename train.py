import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.api.losses import categorical_crossentropy
from keras.api.optimizers import Adam


class FERModelBuilder:

    def __init__(self, dataset_path, training_label, test_label, input_shape, output_model_path):
        self.__dataset_path = dataset_path
        self.__training_label = training_label
        self.__test_label = test_label
        self.__image_shape = input_shape
        self.__output_model_name = output_model_path
        self.__x_train = []
        self.__y_train = []
        self.__x_test = []
        self.__y_test = []

    def start_build(self):
        print("Splitting training and test data...")
        self.__read_split_data()
        print("Splitting complete.\nStarting post processing...")
        self.__post_process_data()
        print("Post processing complete.\nBuilding model...")
        model = self.__build_model()
        print("Model built successfully.\nStarting training...")
        self.__train_model(model)
        print("Model trained successfully.")
        return model

    def __read_split_data(self) -> None:
        data = pd.read_csv(self.__dataset_path)
        for index, row in data.iterrows():
            pixels = np.array(row['pixels'].split(" "), 'float32')
            try:
                if self.__training_label in row['Usage']:
                    self.__x_train.append(pixels)
                    self.__y_train.append(row['emotion'])
                elif self.__test_label in row['Usage']:
                    self.__x_test.append(pixels)
                    self.__y_test.append(row['emotion'])
            except:
                print(f"Error occurred at index: {index} and row: {row}")

    def __post_process_data(self):
        self.__x_train = np.array(self.__x_train, np.float32)
        self.__y_train = np.array(self.__y_train, np.float32)
        self.__x_test = np.array(self.__x_test, np.float32)
        self.__y_test = np.array(self.__y_test, np.float32)

        self.__x_train = normalize_data(self.__x_train)
        self.__x_train = reshape_data(self.__x_train, self.__image_shape)
        self.__x_test = normalize_data(self.__x_test)
        self.__x_test = reshape_data(self.__x_test, self.__image_shape)

    def __build_model(self) -> Sequential:
        model = Sequential()

        # Layer 1
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(self.__x_train.shape[1:])))
        model.add(Activation('relu'))
        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # Layer 2
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # Layer 3
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Fully connected layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(7, activation='softmax'))

        return model

    def __train_model(self, model: Sequential) -> None:
        self.__y_train = to_categorical(self.__y_train, num_classes=7)
        self.__y_test = to_categorical(self.__y_test, num_classes=7)

        model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

        model.fit(self.__x_train,
                  self.__y_train,
                  epochs=30,
                  batch_size=64,
                  validation_data=(self.__x_test, self.__y_test),
                  shuffle=True)

    def save_model(self, model: Sequential):
        model.save(self.__output_model_name)


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def reshape_data(data, image_shape):
    return data.reshape(data.shape[0], *image_shape)
