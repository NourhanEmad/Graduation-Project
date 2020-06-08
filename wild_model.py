import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LSTM, TimeDistributed
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

model = Sequential()
def training_model(X_input, Y_output, num_of_classes, one_hot_encoder, le):
    global model
    in_shape = X_input[0].shape

    # conv 1
    model.add(
        Conv2D(96, kernel_size=3, input_shape=in_shape, strides=(1, 1), activation='relu', padding='same', name="c1"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # conv 2
    model.add(Conv2D(256, kernel_size=3, strides=(2, 2), activation='relu', padding='same', name="c2"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # conv 3
    model.add(Conv2D(512, kernel_size=3, strides=(1, 1), activation='relu', padding='same', name="c3"))

    # conv 4
    model.add(Conv2D(512, kernel_size=3, strides=(1, 1), activation='relu', padding='same', name="c4"))

    # conv 5
    model.add(Conv2D(512, kernel_size=3, strides=(1, 1), activation='relu', padding='same', name="c5"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # FC6
    model.add(TimeDistributed(Flatten()))
    model.add(Dense(512))

    #####################################################################

    print(model.summary())

    # add LSTM
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(Flatten())

    # model.add((Dense(128, activation='relu')))
    model.add((Dense(num_of_classes, activation='softmax')))

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)  # lsa hshof loss eh

    x_train, x_test, y_train, y_test = train_test_split(X_input, Y_output, test_size=0.2, random_state=0,shuffle=True)

    model.fit(x_train,y_train, epochs=10, batch_size=in_shape[0],shuffle=True, verbose=2,validation_data=(x_test, y_test))

    result = model.evaluate(X_test, y_test, verbose=1)
    print("Done testing")

    print("Test loss =", result[0])
    print("Test accuracy =", result[1] * 100)
    model.save_weights("wild_model_weights.h5")

    prediction = model.predict(x_test)
    print("Predicting x_test:")
    inverse_prediction = one_hot_encoder.inverse_transform(prediction.reshape(-1, num_of_classes))
    inverse_prediction = le.inverse_transform(inverse_prediction.astype(int))  # de kalmaat
    inverse_ytest = one_hot_encoder.inverse_transform(y_test.reshape(-1, num_of_classes))
    inverse_ytest = le.inverse_transform(inverse_ytest.astype(int))

    correct = 0
    total = 0
    for i in range(y_test.shape[0]):
        total += 1
        if inverse_prediction[i] == inverse_ytest[i]:
            correct += 1
        print(inverse_prediction[i], "****************", inverse_ytest[i])

    print("#correct:", correct)
    print("total", total)
    print("Wild model overall accuracy", (correct / len(inverse_ytest)) * 100, "%")


def testing(padded_total_words_test, y_labels_test_encoded, one_hot_encoder, le, num_of_classes):
    global model
    prediction = model.predict(padded_total_words_test)

    print("Predicting One video:")
    inverse_prediction = one_hot_encoder.inverse_transform(prediction.reshape(-1, num_of_classes))
    inverse_prediction = le.inverse_transform(inverse_prediction.astype(int))
    inverse_ytest = one_hot_encoder.inverse_transform(y_labels_test_encoded.reshape(-1, num_of_classes))
    inverse_ytest = le.inverse_transform(inverse_ytest.astype(int))
    correct = 0
    for i in range(y_labels_test_encoded.shape[0]):
        if inverse_prediction[i] == inverse_ytest[i]:
            correct += 1
        print(inverse_prediction[i], "##################", inverse_ytest[i])
    print("correct", correct)
    print("total", len(inverse_ytest))
    print("one video test Accuracy", (correct / len(inverse_ytest)) * 100, "%")
