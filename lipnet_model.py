from keras.layers import Activation, GRU, Dropout, TimeDistributed, Dense, Bidirectional, Conv3D, Flatten, MaxPooling3D, \
    ZeroPadding3D, Masking, Embedding
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LSTM
from keras.models import load_model
from keras import optimizers
from keras_preprocessing.sequence import pad_sequences
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import ModelCheckpoint

# model = load_model('lipnet_3D_model_dupplication_10folders_30epoch.h5')
model = Sequential()


def training_model(X_input, Y_output, num_of_classes, one_hot_encoder, le):
    global model
    model = Sequential()
    # print(">>>inside conv3D training_model_FN:")
    # print("Y_output[0]", Y_output[0])
    in_shape = X_input[0].shape

    print("in_shape = ", in_shape)

    #   model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2), input_shape=shape)))
    model.add(Conv3D(32, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu", input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    # momkn a5le el stride (1 1 1)
    # valid, same,
    # kernal size (3 3 3)

    # model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2))))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=(1, 1, 1)))

    # model.add(Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=(1, 1, 1)))

    # model.add(Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(GRU(256, return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(GRU(256, return_sequences=True), merge_mode='concat'))

    model.add(Flatten())
    #     model.add((Dense(128, activation='relu')))
    model.add((Dense(num_of_classes, activation='softmax')))

    print(model.summary())
    print("-------------------------------------------")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # NEW:
    # from scipy.sparse import csr_matrix
    #
    # X_input = csr_matrix(X_input)
    # X_input = X_input.toarray()

    # Y_output = np.asarray(Y_output)
    x_train, x_test, y_train, y_test = train_test_split(X_input, Y_output, test_size=0.2, random_state=0,
                                                        shuffle=True, stratify=Y_output)

    ########3

    # classesInvalidationandNotInTrain = 0
    # setYtest = set()
    # setYtrain = set()
    #
    # for i in range(len(y_test)):
    #     setYtest.add(y_test[i])
    #
    # for i in range(len(y_train)):
    #     setYtrain.add(y_train[i])
    #
    # print("BEFOREsetYtrain", len(y_train))
    # print("setYtrain", len(setYtrain))
    # print("setYtest", len(setYtest))
    # x=len(setYtest)
    #
    # # setYtest = np.asarray(setYtest)
    #
    # for i in range(x):
    #     z = setYtest.pop()
    #     print(z)
    #     if (z in setYtrain == False):
    #         classesInvalidationandNotInTrain += 1
    #
    # print("classesInvalidationandNotInTrain", classesInvalidationandNotInTrain)
    ###########3
    y_labels_encoded = le.fit_transform(y_train)
    y_labels_encoded = np.reshape(y_labels_encoded, (-1, 1))

    y_train = one_hot_encoder.fit_transform(y_labels_encoded)  # .toarray()
    ###########
    y_labels_encoded = le.fit_transform(y_test)
    y_labels_encoded = np.reshape(y_labels_encoded, (-1, 1))

    y_test = one_hot_encoder.fit_transform(y_labels_encoded)  # .toarray()
    print("------------BEEEEEEEEEEEEEEEEBB--------------")

    # print("shape x_train before validation", x_train.shape)
    # print("shape y_train before validation", y_train.shape)

    #    x_train, x_val,  y_train, y_val  = train_test_split(x_train, y_train , test_size=0.1, random_state=0, shuffle=True, stratify=Y_output)

    # X_input--> X_train, Y_output-->y_train
    #    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=in_shape[0], verbose=1)

    # print(">>>>>>x_test[0].shape", x_test[0].shape)
    # checkpoint
    filepath = "model_weights.h5"
    # model.load_weights(filepath)
    # print("weights loaded")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train, shuffle=True, epochs=20, validation_data=(x_test, y_test),
                        batch_size=in_shape[0], verbose=2, callbacks=callbacks_list)
    model.save_weights("lipnet_model_weights.h5")

    #    print("history", history.history)

    # NEW:
    evaluation_result = model.evaluate(x_test, y_test, verbose=1)
    # print("Evaluation conv3D model loss =", evaluation_result[0])
    # print("Evaluation conv3D model accuracy =", evaluation_result[1] * 100)

    prediction = model.predict(x_test)
    # print("prediction[0]:", prediction[0].shape)
    # print(prediction[0])
    # print("type prediction", type(prediction))
    # print("shape prediction", prediction.shape)
    # print("y test shape", y_test.shape)
    # prediction[0, :] = np.where(prediction[0, :] == max(prediction[0, :]), 1, 0)

    print("Predicting x_test:")
    inverse_prediction = one_hot_encoder.inverse_transform(prediction.reshape(-1, num_of_classes))
    # print("inverse after one hot encoding", inverse_prediction)
    inverse_prediction = le.inverse_transform(inverse_prediction.astype(int))  # de kalmaat

    inverse_ytest = one_hot_encoder.inverse_transform(y_test.reshape(-1, num_of_classes))
    # print("inverse after one hot encoding", inverse_ytest)
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
    # print("conv3D Overall Accuracy", (correct / y_test.shape[0]) * 100, "%")
    print("conv3D Overall Accuracy", (correct / len(inverse_ytest)) * 100, "%")

    # print("done fitting CONV3D")


#    model.save("lipnet_3D_model_dupplication_10folders_30epoch.h5")


def testing(padded_total_words_test, y_labels_test_encoded, one_hot_encoder, le, num_of_classes):
    global model
    prediction = model.predict(padded_total_words_test)

    print("Predicting One video:")
    # print("inverse transform:")
    inverse_prediction = one_hot_encoder.inverse_transform(prediction.reshape(-1, num_of_classes))

    # print("inverse after one hot encoding", inverse_prediction)
    inverse_prediction = le.inverse_transform(inverse_prediction.astype(int))
    # print("inverse after transform",inverse_prediction)

    inverse_ytest = one_hot_encoder.inverse_transform(y_labels_test_encoded.reshape(-1, num_of_classes))
    # print("inverse after one hot encoding", inverse_ytest)
    inverse_ytest = le.inverse_transform(inverse_ytest.astype(int))
    correct = 0
    for i in range(y_labels_test_encoded.shape[0]):
        if inverse_prediction[i] == inverse_ytest[i]:
            correct += 1
        print(inverse_prediction[i], "##################", inverse_ytest[i])
    print("correct", correct)
    print("total", len(inverse_ytest))
    print("one video test Accuracy", (correct / len(inverse_ytest)) * 100, "%")
