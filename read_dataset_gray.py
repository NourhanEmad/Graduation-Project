import cv2
import numpy as np
import os
import math
import pickle
from keras.layers import Activation, GRU, Dropout, TimeDistributed, Dense, Bidirectional, Conv3D, Flatten, MaxPooling3D, \
    ZeroPadding3D, Masking, Embedding
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LSTM
from keras import optimizers
from keras_preprocessing.sequence import pad_sequences
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
import mouth_extraction
import wild_model
import lipnet_model
from keras.callbacks import ModelCheckpoint

total_words = []
total_words_gray = []
new_total_words = []
y_labels = []
# y_labels_test = []
num_of_classes = 0
count_classes = set()
num_of_classes_test = 0
total_lines = 1
detected_mouthes = 0
no_mouthes = 0
maxframes = 0

y_label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()


def get_max_num_of_frames():
    global maxframes
    maxframes = 0
    for word in total_words:
        frames_count = len(word)
        maxframes = max(maxframes, frames_count)
    return maxframes


def duplicate_frames():
    # maxframes = get_max_num_of_frames()
    global maxframes
    global new_total_words
    for wordlist in total_words:
        wordframes = len(wordlist)
        new_wordlist = []
        new_wordlist = np.asarray(new_wordlist)
        numberOfDuplication = maxframes // wordframes  # // msh bygeb ele b3d el point

        for i in range(len(wordlist)):  # for each frame
            for j in range(numberOfDuplication):
                new_wordlist = np.append(new_wordlist, wordlist[i])
                # for gray frames
                new_wordlist = np.reshape(new_wordlist, (-1, wordlist[i].shape[0], wordlist[i].shape[1]))
                # for coloured frames
                # new_wordlist = np.reshape(new_wordlist,
                #                           (-1, wordlist[i].shape[0], wordlist[i].shape[1], wordlist[i].shape[2]))

        if maxframes % wordframes != 0:  # lw etb2a ksor (ba2y el esma) bkml b a5r frame
            for j in range(maxframes % wordframes):
                new_wordlist = np.append(new_wordlist,
                                         wordlist[len(wordlist) - 1])  # append last element multiple time
                # for grey frames
                new_wordlist = np.reshape(new_wordlist, (
                -1, wordlist[len(wordlist) - 1].shape[0], wordlist[len(wordlist) - 1].shape[1]))  # for grey frames
                # for coloured frames
                # new_wordlist = np.reshape(new_wordlist,
                #                           (-1, wordlist[len(wordlist) - 1].shape[0],
                #                            wordlist[len(wordlist) - 1].shape[1], wordlist[len(wordlist) - 1].shape[2]))

        # for grey frames
        new_wordlist = np.reshape(new_wordlist, (-1, new_wordlist.shape[1], new_wordlist.shape[2]))
        # for coloured frames
        # new_wordlist = np.reshape(new_wordlist,
        #                           (-1, new_wordlist.shape[1], new_wordlist.shape[2], new_wordlist.shape[3]))

        new_total_words.append(new_wordlist)
    # return


# returns frames: training and testing
def get_words(file_name, times):  # file name of video-----times list of tuple(start,end)
    global total_words
    global total_lines
    global total_words_gray
    num_of_lines_in_one_file = 0

    cap = cv2.VideoCapture(file_name)
    print("video name________-", file_name)
    for item in times:
        num_of_lines_in_one_file += 1
        # print("LINE #", total_lines)
        total_lines += 1
        word = []
        word = np.asarray(word)
        word_gray = []
        word_gray = np.asarray(word_gray)

        start_time = item[0]
        end_time = item[1]
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1 / fps

        start_frame_no = math.ceil(start_time / frame_duration)
        end_frame_no = math.ceil(end_time / frame_duration)

        # get a range of frames from the video:
        while start_frame_no <= end_frame_no:
            cap.set(1, start_frame_no)  # (1 --> frame_no) # change the current frame to a specified frame
            ret, colored_frame = cap.read()
            # colored_frame = cv2.resize(colored_frame,(112,112))
            # colored_frame.reshape(3,112,112)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # print("WIDTH", width)
            mouth = mouth_extraction.Mouth_extraction(colored_frame).extract_mouth_dnn(int(width), int(
                height))  # bnb3tlo frame wa7d

            ### gray frames ###
            gray_frame = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            word_gray = np.append(word_gray, gray_frame)
            word_gray = np.reshape(word_gray, (-1, gray_frame.shape[0], gray_frame.shape[1]))  # , frame.shape[2]))
            ### gray frames ###

            ### colored frames ###
            word = np.append(word, mouth)  # list of colered frames for one word
            # print("word shape after appending one frame", word.shape)
            word = np.reshape(word, (-1, mouth.shape[0], mouth.shape[1], mouth.shape[2]))
            ### colored frames ###

            start_frame_no += 1
        ### for gray frames ###
        word_gray = np.reshape(word_gray, (-1, word.shape[1], word.shape[2]))  # 4 150 150

        ### for colored frames ###
        word = np.reshape(word, (-1, word.shape[1], word.shape[2], word.shape[3]))  # 4 150 150 3

        # print("complete Word shape", word.shape)

        total_words.append(word)
        total_words_gray.append(word_gray)
        # print("total_words len", len(total_words))
        # print("total_words len[0]", len(total_words[0]))

        # print("--------------------------")

    # print("num_of_lines_in_one_file", num_of_lines_in_one_file)

    cap.release()

    # Closes all the frames
    # cv2.destroyAllWindows()

    return


def read_video_txt_file(file_name):
    global y_labels
    global num_of_classes
    global count_classes
    count_classes = set()

    # print("in read_files", file_name)
    begin = False
    one_video_time_intervals = []

    f = open(file_name, "r")
    whole_file = f.readlines()

    for line in whole_file:

        line_content = line.split(" ")
        if begin:
            y_labels.append(line_content[0].lower())  # shayla kol el words bl occurences
            count_classes.add(line_content[0].lower())  # shayla el word el unique

            t = float(line_content[1]), float(line_content[2])  # start time --- end time
            one_video_time_intervals.insert(-1, t)

        if line_content[0].lower() == 'word':
            begin = True

    num_of_classes = len(count_classes)

    return one_video_time_intervals


def load_data():
    global total_words
    global total_words_gray
    global y_labels
    global num_of_classes

    if os.path.exists('complete_colored_frames_samples'):

        # print("Training data exists :)")
        # print("Loading from drive...")

        with open('complete_colored_frames_samples', 'rb') as f:
            total_words = pickle.load(f)
        print("first colored frames ", len(total_words))
        # with open('complete_colored_frames10_2', 'rb') as f:
        #   total_words.extend(pickle.load(f))
        # print("second colored frames ", len(total_words))
        # with open('complete_colored_frames10_3', 'rb') as f:
        #   total_words.extend(pickle.load(f))
        # print("third colored frames ", len(total_words))

        # with open('complete_colored_frames10', 'rb') as f:
        #   total_words = pickle.load(f)

        with open('complete_gray_frames_samples', 'rb') as f:
            total_words_gray = pickle.load(f)

        #        with open('labels', 'rb') as f:
        #            y_labels = pickle.load(f)

        with open('labels_samples', 'rb') as f:
            y_labels = pickle.load(f)
        # with open('labels10_2', 'rb') as f:
        # y_labels.extend(pickle.load(f))
        # with open('labels10_3', 'rb') as f:
        # y_labels.extend(pickle.load(f))

        with open('num_of_classes_samples.txt', 'r') as f:
            num_of_classes = int(f.read())
        # with open('num_of_classes10_2.txt', 'r') as f:
        #   num_of_classes += int(f.read())
        # with open('num_of_classes10_3.txt', 'r') as f:
        #    num_of_classes += int(f.read())

        print("Loaded:")
        print("len total_word", len(total_words))
        print("len y_labels", len(y_labels))
        print("number of frames for any word", len(total_words[11]))

    else:
        print("doesn't exist, reading data")

        Dir = "pretrain20"
        for subfolder_name in os.listdir(Dir):
            subfolder = os.path.join(Dir, subfolder_name)  # 5588894046 contains kza txt files, kza mp4 files

            unique_file_names = set()
            for file in os.listdir(subfolder):

                file_name = file.split(".")[0]

                if file_name not in unique_file_names:
                    # print("NOT in unique")
                    unique_file_names.add(file_name)

                    txt_file = file_name + ".txt"
                    txt_file = os.path.join(subfolder, txt_file)

                    one_video_time_intervals = read_video_txt_file(
                        txt_file)  # returns list of y's & list of times for each video

                    # print("After read File fn:")
                    # print(y_labels)
                    print("y_labels len", len(y_labels))
                    # print("y_labels[0] len", len(y_labels[0]), y_labels[0])

                    mp4_file = file_name + ".mp4"
                    mp4_file = os.path.join(subfolder, mp4_file)

                    get_words(mp4_file, one_video_time_intervals)

                    # print("after reading a whole file, len total_words", len(total_words))
                    # print("****************************************")

        with open('complete_colored_frames_samples', 'wb') as f:
            pickle.dump(total_words, f)

        with open('complete_gray_frames_samples', 'wb') as f:
            pickle.dump(total_words_gray, f)

        with open('labels_samples', 'wb') as f:
            pickle.dump(y_labels, f)

        with open('num_of_classes_samples.txt', 'w') as f:
            f.write(str(num_of_classes))


def main(mode):
    global one_hot_encoder
    global y_label_encoder
    global new_total_words
    global total_words
    global y_labels
    global num_of_classes
    global num_of_classes_test
    global maxframes

    load_data()

    ''' ####################################### edit 17/5 ######################################
     folders_names = ['complete_colored_frames10_2','complete_colored_frames','complete_colored_frames10_3']
     num_of_classes_folders_names = ['num_of_classes10_2.txt','num_of_classes.txt','num_of_classes10_3.txt']
     labels_folders_names = ['labels10_2','labels', 'labels10_3']
     for i in range(3):
    
       total_words = []
       new_total_words = []
       y_labels = []
       num_of_classes = 0
       num_of_classes_test = 0
       maxframes = 0
       y_label_encoder = LabelEncoder()
       one_hot_encoder = OneHotEncoder()
    
       new_total_words = []
       print("in the loop, i = ",i)
       with open(folders_names[i], 'rb') as f:
             total_words = pickle.load(f)
    
       with open(labels_folders_names[i], 'rb') as f:
             y_labels = pickle.load(f)
    
       with open(num_of_classes_folders_names[i], 'r') as f:
             num_of_classes = int(f.read())'''

    if mode == "duplication":
        maxmm = get_max_num_of_frames()
        print("get_max_num_of_frames", maxmm)
        duplicate_frames()

    else:
        new_total_words = pad_sequences(total_words_gray, padding='pre')
    print("y_labels", len(y_labels))
    y_labels_encoded = y_label_encoder.fit_transform(y_labels)
    y_labels_encoded = np.reshape(y_labels_encoded, (-1, 1))

    y_labels_encoded = one_hot_encoder.fit_transform(y_labels_encoded)  # .toarray()
    new_total_words = np.asarray(new_total_words)
    print("Conv3D model:")
    # lipnet_model.training_model(new_total_words, y_labels_encoded,
    #                                 num_of_classes, one_hot_encoder,
    #                                 y_label_encoder)  # added last parameter

    #      num_of_classes_test = num_of_classes
    #      testing(mode)

    #    print("----------------------")

    #    print("**Wild Model:")
    wild_model.training_model(new_total_words, y_labels_encoded, num_of_classes, one_hot_encoder, y_label_encoder)


def testing(mode):
    total_words_test = []
    global y_labels_test
    global y_labels
    global new_total_words
    y_labels.clear()
    global num_of_classes_test
    global maxframes

    time = read_video_txt_file("00020.txt")
    y_labels_test = y_labels.copy()

    total_words_gray = []
    get_words("00020.mp4", time)  # btmla total_words
    total_words_test = total_words_gray.copy()

    y_labels_test_encoded = y_label_encoder.transform(y_labels_test)
    y_labels_test_encoded = np.reshape(y_labels_test_encoded, (-1, 1))

    y_labels_test_encoded = one_hot_encoder.transform(y_labels_test_encoded)  # .toarray()

    new_total_words = []

    if mode == "duplication":
        # maxmm = get_max_num_of_frames()
        # print("maxmm", maxmm)
        duplicate_frames()  # btmla new total words
    else:
        new_total_words = pad_sequences(total_words_test, padding='pre')

    new_total_words = np.asarray(new_total_words)
    # lipnet_model.testing(new_total_words, y_labels_test_encoded, one_hot_encoder,
    #                          y_label_encoder, num_of_classes_test)
    wild_model.testing(new_total_words, y_labels_test_encoded, one_hot_encoder,
                       y_label_encoder, num_of_classes_test)
    return


num_of_classes = 0
count_classes = 0
mode = "duplication"
# mode = "padding"
main(mode)
num_of_classes_test = num_of_classes
testing(mode)
