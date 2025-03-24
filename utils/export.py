import os
import copy
import pickle as pkl
import numpy as np
import pandas as pd

import torch

from utils.midi_processing import convert_midi_to_piano_roll
from dataloader import PianoRollDataset

def export_features(data_path, output_path, model_name = 'transformer_maestro_test', out_fs = 100, timing_correction = False):
    '''
    description: export time-resolved features from a trained model. The features are computed from the model's predictions on the input data.

    parameter: data_path, path to the MIDI file from which to retrieve the model's features
    parameter: output_path, path where the features will be stored
    parameter: model_name, name of the model folder in ./versions/
    parameter: out_fs, sampling frequency of the time resolved features. If out_fs is higher than the model's sampling frequency,
    the features will be interpolated.
    parameter: timing_correction. 
        If False, the time resolved features will remain temporally aligned with the piano rolls processed by the model. This will cause 
        small temporal shifts between the features and the actual note onsets, proportional to the model's sampling frequency. 
        If True, the time resolved features will be re-aligned with the exact note onsets, but this may slightly distord the 
        values (linear interpolation). This is useful when precise timing information is required.

    features list:
        - surprise: Binary Cross Entropy (BCE) between the model's predictions and the target values
        - surprise_max: maximum BCE across all simulatenous notes
        - surprise_scaled: BCE scaled by the number of simultaneous notes in the target
        - surprise_positive: positive part of the BCE
        - surprise_positive_max: maximum positive part of the BCE across all simulatenous notes
        - surprise_positive_scaled: positive part of the BCE scaled by the number of simultaneous notes in the target
        - surprise_negative: negative part of the BCE
        - surprise_negative_max: maximum negative part of the BCE across all absent notes
        - surprise_negative_scaled: negative part of the BCE scaled by the number of absent notes in the target
        - uncertainty: entropy of the model's predictions. The probabilities are normalized (sum = 1) before computing the entropy.
        - predicted_density: predicted note density (sum of probabilities)

    files:
        - features.csv: contains a summary of the features of all midi files in a single table. Each feature is summed over time, either
                        across all timesteps ('continuous_xxx' features) or only at the timesteps with note onsets ('onsets_xxx' features). In
                        addition, the table contains the number of notes, events (group of simulatenous notes) and the duration of each file.
        - features_over_time: folder containing the time-resolved features for each midi file. Each file contain all features, with the 
                        addition of a time axis and a binary mask for the note onsets.
    '''

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #create output folder
    output_path = output_path + '/features_' + model_name + '/'
    os.makedirs(output_path, exist_ok=True)
    #create tmp folder
    os.makedirs('./tmp_export', exist_ok=True)

    #model import
    model = torch.load('./versions/'+ model_name +'/model.pt', weights_only=False, map_location=device) #model class must be imported
    model.device = device
    info = torch.load('./versions/' + model_name + '/info.pt', map_location=device)
    model_type = info['model_type']
    fs = info['fs']
    ons_value = info['ons_value']
    sus_value = info['sus_value']
    padding_value = info['padding_value']
    output_size = info['output_size']
    if model_type == 'transformer':
        max_seq_length = info['max_seq_length']

    #convert midi to piano roll
    convert_midi_to_piano_roll(data_path = data_path, out_dir = './tmp_export/', file_name = None,
                               fs = fs, pedal_threshold = 64, save_midi_timings = True)
    
    #load midi files
    dataset = PianoRollDataset(data_path = './tmp_export/', 
               dataset_fs = fs, model_fs = fs, 
               ons_value = ons_value, sus_value = sus_value, 
               padding_value = padding_value,
               source_length = None,
               use_transposition = False,
               preload = True, device = device, dtype = torch.float32)
    
    #get model predictions
    output_list = []
    target_list = []
    file_list = []
    model.eval()
    for i_sample in range(dataset.__len__()):
        sample, file_name = dataset.__getitem__(i_sample)
        print('Computing predictions for sample ', i_sample+1, '/', dataset.__len__(), ': ', file_name)

        #get model's input and target
        src = sample.clone()[: -1, :]
        tgt = sample.clone()[1:, :]

        #remove sustain values in targets
        tgt[tgt == sus_value] = 0
        tgt[tgt == ons_value] = 1

        #add initial silence (10 seconds) to avoid starting bias
        silence_length = 10*fs
        src = torch.cat((torch.zeros((silence_length,88)), src),dim = 0)
        tgt = torch.cat((torch.zeros((silence_length,88)), tgt),dim = 0)

        #get model predictions
        src = src.to(device)
        tgt = tgt.to(device)
        model.eval()
        if model_type == 'transformer':
            output = torch.zeros((tgt.size(0), 88), device = device)
            if src.size(0) - max_seq_length > 0:
                for i in range(src.size(0) - max_seq_length):
                    S = src[i:i+max_seq_length, :]
                    S = S.unsqueeze(0)
                    with torch.no_grad():
                        out = model(S)
                        out = out.squeeze(0)
                    if i == 0:
                        output[:max_seq_length, :] = out
                    else:
                        output[i+max_seq_length-1, :] = out[-1, :]
            else:
                S = src.unsqueeze(0)
                with torch.no_grad():
                    out = model(S)
                    out = out.squeeze(0)
                output = out
        elif model_type == 'lstm':
            S = src.unsqueeze(0)
            (h,c) = model.init_hidden(batch_size = 1)
            with torch.no_grad():
                out, (h,c) = model(S,(h,c))
            output = out.squeeze(0)
        
        #remove initial silence
        tgt = tgt[silence_length -1:, :]
        output = output[silence_length -1:, :] #keep the prediction about the first timestep of the sequence (after the silence)

        #add to list
        output_list.append(output.cpu().numpy().T)
        target_list.append(tgt.cpu().numpy().T)
        file_list.append(file_name.split('.')[0])
    
    #compute features
    features = {}
    for i_sample in range(dataset.__len__()):
        file_name = file_list[i_sample]
        features[file_name] = {}
        print('Computing features for sample ', i_sample+1, '/', dataset.__len__(), ': ', file_name)

        out = output_list[i_sample]
        tgt = target_list[i_sample]
        
        features[file_name]['model_input_timing'] = np.where(np.sum(tgt,axis=0) == 0, 0, 1)

        note_density = np.sum(tgt,axis=0)
        surprise = np.where(tgt == 1,-np.log2(out), -np.log2(1-out)) #overall surprise (binary cross entropy)
        features[file_name]['surprise'] = np.sum(surprise,axis = 0)
        features[file_name]['surprise_max'] = np.max(surprise,axis = 0)
        features[file_name]['surprise_scaled'] = np.sum(surprise,axis = 0) / np.where(note_density == 0, 1, note_density)

        surprise_positive = np.where(tgt == 1,-np.log2(out),0) #positive part of the binary cross entropy
        features[file_name]['surprise_positive'] = np.sum(surprise_positive,axis = 0)
        features[file_name]['surprise_positive_max'] = np.max(surprise_positive,axis = 0)
        features[file_name]['surprise_positive_scaled'] = np.sum(surprise_positive,axis = 0) / np.where(note_density == 0, 1, note_density)

        surprise_negative = np.where(tgt == 0,-np.log2(1-out),0) #negative part of the binary cross entropy
        features[file_name]['surprise_negative'] = np.sum(surprise_negative,axis = 0)
        features[file_name]['surprise_negative_max'] = np.max(surprise_negative,axis = 0)
        features[file_name]['surprise_negative_scaled'] = np.sum(surprise_negative,axis = 0) / np.where(note_density == 0, output_size, output_size-note_density)

        out_normalized = out / np.sum(out, axis = 0)
        features[file_name]['uncertainty'] = np.sum(-np.log2(out_normalized)*out_normalized,axis = 0) #uncertainty
        features[file_name]['predicted_density'] = np.sum(out, axis = 0)                                 #sum of probabilities
    
    #create & save features summary
    summary = {}
    for i_sample in range(dataset.__len__()):
        file_name = file_list[i_sample]

        mask_onset = features[file_name]['model_input_timing'].astype(bool)
        summary[file_name] = {}
        summary[file_name]['file_name'] = file_name
        summary[file_name]['n_events'] = np.sum(mask_onset)
        summary[file_name]['n_notes'] = target_list[i_sample].sum().astype(int)
        summary[file_name]['duration'] = target_list[i_sample].shape[1] / fs

        for label in features[file_name].keys():
            if label != 'model_input_timing':
                summary[file_name]['continous_' + label] = np.sum(features[file_name][label])
        for label in features[file_name].keys():
            if label != 'model_input_timing':
                summary[file_name]['onsets_' + label] = np.sum(features[file_name][label][mask_onset])
    df_summary = pd.DataFrame(summary).T
    df_summary.to_csv(output_path + 'features.csv', sep = ',', index = False)

    #upsample time-resolved features & apply timing correction
    with open('./tmp_export/timings.pkl', 'rb') as f:
        timings = pkl.load(f)
    for i_sample in range(dataset.__len__()):
        file_name = file_list[i_sample]
        out = output_list[i_sample]
        onsets = np.sort(timings[file_name])

        #original time axis
        original_time_axis = np.linspace(0,(out.shape[1]-1)*(1/fs),num=out.shape[1])

        #create resampled time axis
        new_time_axis = np.linspace(0, original_time_axis[-1], num = int(original_time_axis[-1]*out_fs +1))
        features[file_name]['time_axis'] = new_time_axis
        
        #find index of timesteps that are the closest to real onset timings, for timing correction
        if timing_correction: 
            onsets_idx = []
            for onset in onsets:
                onsets_idx.append((np.abs((original_time_axis) - onset)).argmin())
        
        #shift timesteps of the time axis to the nearest onset timings, for timing correction
        if timing_correction:
            corrected_time_axis = np.copy(original_time_axis)
            for i in range(len(onsets_idx)):
                if (onsets_idx == onsets_idx[i]).sum() > 1: #if multiple onsets in the same timestep, timing will be set to the average
                    corrected_time_axis[onsets_idx[i]] = np.mean(onsets[np.where(onsets_idx == onsets_idx[i])]) 
                else:
                    corrected_time_axis[onsets_idx[i]] = onsets[i]

        #linear interpolation for continuous features
        continuous_features = ['surprise','surprise_max','surprise_scaled',
                                'surprise_negative','surprise_negative_max','surprise_negative_scaled',
                                'uncertainty','predicted_density']
        for feature in continuous_features:
            if timing_correction:
                features[file_name][feature] = np.interp(new_time_axis, corrected_time_axis, features[file_name][feature])
            else:
                features[file_name][feature] = np.interp(new_time_axis, original_time_axis, features[file_name][feature])

        #nearest neighbor for discrete features
        features[file_name]['onset'] = copy.copy(features[file_name]['model_input_timing']) #add onset binary mask to be upsampled
        discrete_features = ['onset','surprise_positive','surprise_positive_max','surprise_positive_scaled']
        target_idx = np.where(features[file_name]['model_input_timing'])[0] #retrieve index of timesteps with notes in the original time axis
        if timing_correction: #handle occasional missing notes in the piano roll, when compared to midi files
            if len(target_idx) < len(onsets_idx): 
                tmp = [] 
                for i in range(len(onsets_idx)):
                    idx = np.argmin(np.abs(target_idx - onsets_idx[i]))
                    tmp.append(target_idx[idx])
                target_idx = tmp
        if timing_correction: #find index of timesteps to place the onsets, in the new time axis
            new_targets_idx = []
            for onset in onsets:
                new_targets_idx.append((np.abs((new_time_axis) - onset)).argmin()) 
        else:
            new_targets_idx = (target_idx * (out_fs/fs)).astype(int)
        for feature in discrete_features: #add values at the right timesteps
            tmp = np.zeros(new_time_axis.shape)
            tmp[new_targets_idx] = features[file_name][feature][target_idx] 
            features[file_name][feature] = tmp
        del features[file_name]['model_input_timing'] #remove onset binary mask with old frequency sampling
    
    #save time-resolved features
    if not os.path.exists(output_path + '/features_over_time/'):
        os.makedirs(output_path + '/features_over_time/')
    for i_sample in range(dataset.__len__()):
        file_name = file_list[i_sample]
        out = np.stack([features[file_name][feature] for feature in features[file_name].keys()],axis=1)
        np.savetxt(output_path + '/features_over_time/' + file_name + '.csv', out, delimiter = ',', header = ','.join(features[file_name].keys()), comments = '')

    #clean tmp folder with its content
    for file in os.listdir('./tmp_export/'):
        os.remove('./tmp_export/' + file)
    os.rmdir('./tmp_export/')