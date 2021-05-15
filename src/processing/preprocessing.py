import os, sys
import numpy as np
import librosa
import pickle
import random
from src.utils import utility_functions as uf
from src.processing import feature_extraction as fe

'''
Process the unzipped dataset folders and output numpy matrices (.pkl files)
containing the pre-processed data for task1 and task2, separately.
Separate training, validation and test matrices are saved.
Command line inputs define which task to process and its parameters.
'''

sound_classes_dict_task2 = {'Chink_and_clink':0,
                           'Computer_keyboard':1,
                           'Cupboard_open_or_close':2,
                           'Drawer_open_or_close':3,
                           'Female_speech_and_woman_speaking':4,
                           'Finger_snapping':5,
                           'Keys_jangling':6,
                           'Knock':7,
                           'Laughter':8,
                           'Male_speech_and_man_speaking':9,
                           'Printer':10,
                           'Scissors':11,
                           'Telephone':12,
                           'Writing':13}


def preprocessing_task2(input_path,
                        output_path,
                        train_val_split = 0.8,
                        num_mics = 1,
                        num_data = 100,
                        frame_len = 100,
                        stft_nperseg = 512,
                        stft_noverlap = 112,
                        stft_window = 'hamming',
                        output_phase = False,
                        predictors_len_segment = None,
                        target_len_segment = None,
                        segment_overlap = None,
                        ov_subsets = ["ov1", "ov2", "ov3"],
                        no_overlaps = False,
                        fs=32000, 
                        n_fft=2048, 
                        hop_length=400, 
                        n_mel_bands=256, 
                        frame_length=2049):
    '''
    predictors output: ambisonics stft
                       Matrix shape: -x data points
                                     - num freqency bins
                                     - num time frames
    target output: matrix containing all active sounds and their position at each
                   100msec frame.
                   Matrix shape: -x data points
                                 -600: frames
                                 -168: 14 (clases) * 3 (max simultaneous sounds per frame)
                                       concatenated to 14 (classes) * 3 (max simultaneous sounds per frame) * 3 (xyz coordinates)
    '''
    sr_task2 = 32000
    sound_classes=['Chink_and_clink','Computer_keyboard','Cupboard_open_or_close',
             'Drawer_open_or_close','Female_speech_and_woman_speaking',
             'Finger_snapping','Keys_jangling','Knock',
             'Laughter','Male_speech_and_man_speaking',
             'Printer','Scissors','Telephone','Writing']
    file_size=60.0
    max_label_distance = 2.  #maximum xyz value (serves for normalization)

    def process_folder(folder):
        print ('Processing ' + folder.split('/')[-1] + ' folder...')
        predictors = []
        target = []
        data_path = os.path.join(folder, 'data')
        labels_path = os.path.join(folder, 'labels')

        data = os.listdir(data_path)
        data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']
        count = 0
        for sound in data:
            ov_set = sound.split('_')[-3]
            if ov_set in ov_subsets:  #if data point is in the desired subsets ov
                target_name = 'label_' + sound.replace('_A', '').replace('.wav', '.csv')
                sound_path = os.path.join(data_path, sound)
                target_path = os.path.join(data_path, target_name)
                target_path = '/'.join((target_path.split('/')[:-2] + ['labels'] + [target_path.split('/')[-1]]))  #change data with labels
                #target_path = target_path.replace('data', 'labels')  #old
                samples, sr = librosa.load(sound_path, sr_task2, mono=False)
                if num_mics == 2:  # if both ambisonics mics are wanted
                    #stack the additional 4 channels to get a (8, samples) shape
                    B_sound_path = sound_path[:-5] + 'B' +  sound_path[-4:]  #change A with B
                    #B_sound_path = sound_path.replace('A', 'B')  old
                    samples_B, sr = librosa.load(B_sound_path, sr_task2, mono=False)
                    samples = np.concatenate((samples,samples_B), axis=-2)

                #compute features
                # STFT
                stft = uf.spectrum_fast(samples, 
                                        nperseg=stft_nperseg,
                                        noverlap=stft_noverlap,
                                        window=stft_window,
                                        output_phase=output_phase)
                
                # logmel_IV
                
                features = fe.get_logmel_IV(samples, 
                                            fs=fs, 
                                            n_fft=n_fft, 
                                            hop_length=hop_length, 
                                            n_mel_bands=n_mel_bands, 
                                            frame_length=frame_length)
                
                features = features[:,:,:-1]

                #stft = np.reshape(samples, (samples.shape[1], samples.shape[0],
                #                     samples.shape[2]))


                #compute matrix label
                label = uf.csv_to_matrix_task2(target_path, sound_classes_dict_task2,
                                               dur=60, step=frame_len/1000., max_loc_value=2.,
                                               no_overlaps=no_overlaps)  #eric func

                #label = uf.get_label_task2(target_path,0.1,file_size,sr_task2,          #giuseppe func
                #                        sound_classes,int(file_size/(frame_len/1000.)),
                #                        max_label_distance)


                #segment into shorter frames
                if predictors_len_segment is not None and target_len_segment is not None:
                    #segment longer file to shorter frames
                    #not padding if segmenting to avoid silence frames
                    predictors_cuts, target_cuts = uf.segment_task2(features, label, predictors_len_segment=predictors_len_segment,
                                                    target_len_segment=target_len_segment, overlap=segment_overlap)

                    for i in range(len(predictors_cuts)):
                        predictors.append(predictors_cuts[i])
                        target.append(target_cuts[i])
                        #print (predictors_cuts[i].shape, target_cuts[i].shape)
                else:

                    predictors.append(features)
                    target.append(label)

                #print (samples.shape, np.max(label), np.min(label))

                count += 1
                if num_data is not None and count >= num_data:
                    break


        return predictors, target

    train_folder = os.path.join(input_path, 'L3DAS_Task2_train')
    test_folder = os.path.join(input_path, 'L3DAS_Task2_dev')

    predictors_train, target_train = process_folder(train_folder)
    predictors_test, target_test = process_folder(test_folder)

    predictors_test = np.array(predictors_test)
    target_test = np.array(target_test)
    #print (predictors_test.shape, target_test.shape)

    #split train set into train and development
    split_point = int(len(predictors_train) * train_val_split)
    predictors_training = predictors_train[:split_point]    #attention: changed training names
    target_training = target_train[:split_point]
    predictors_validation = predictors_train[split_point:]
    target_validation = target_train[split_point:]

    #save numpy matrices into pickle files
    print ('Saving files')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path,'task2_predictors_train.pkl'), 'wb') as f:
        pickle.dump(predictors_training, f, protocol=4)
    with open(os.path.join(output_path,'task2_predictors_validation.pkl'), 'wb') as f:
        pickle.dump(predictors_validation, f, protocol=4)
    with open(os.path.join(output_path,'task2_predictors_test.pkl'), 'wb') as f:
        pickle.dump(predictors_test, f, protocol=4)
    with open(os.path.join(output_path,'task2_target_train.pkl'), 'wb') as f:
        pickle.dump(target_training, f, protocol=4)
    with open(os.path.join(output_path,'task2_target_validation.pkl'), 'wb') as f:
        pickle.dump(target_validation, f, protocol=4)
    with open(os.path.join(output_path,'task2_target_test.pkl'), 'wb') as f:
        pickle.dump(target_test, f, protocol=4)

    print ('Matrices successfully saved')
    print ('Training set shape: ', np.array(predictors_training).shape, np.array(target_training).shape)
    print ('Validation set shape: ', np.array(predictors_validation).shape, np.array(target_validation).shape)
    print ('Test set shape: ', np.array(predictors_test).shape, np.array(target_test).shape)


