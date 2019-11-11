import os
import pickle
import json
import numpy as np


def load_calib_file(calib_file):
    assert os.path.isfile(calib_file)
    with open(calib_file) as f:
        calib = json.load(f)
    for key in calib:
        if type(calib[key]) == list:
            calib[key] = np.array(calib[key])
    return calib


"""
#################################################################
Panoptic A4
#################################################################
"""
# run in Python 3

root = '/media/posefs0c/panopticdb/a4/'
sample_list = os.path.join(root, 'sample_list.pkl')

with open(sample_list, 'rb') as f:
    df = pickle.load(f)

# collect hand data
if os.path.isfile('./a4_collected.pkl'):
    print('A4 collection file exists.')
else:
    training_data = []
    testing_data = []
    for seqName, seq_samples in df.items():
        i = 0
        for hvframe, frame_dict in seq_samples.items():
            i += 1
            hv, frame_str = hvframe
            print('collecting data: {} {}/{}'.format(seqName, i, len(seq_samples)))
            person3df = os.path.join(root, 'annot_{}_3d'.format(hv), seqName, 'Recon3D_{0}{1}.json'.format(hv, frame_str))
            with open(person3df) as f:
                print(person3df)
                person3d = json.load(f)

            map_id = {}
            for person_data in person3d:
                pid = person_data['id']
                if pid == -1:
                    continue
                person_dict = {'seqName': seqName, 'frame_str': frame_str, 'id': pid}
                body_dict = {'landmarks': person_data['body']['landmarks'], '2D': {}}
                person_dict['body'] = body_dict
                if 'subjectsWithValidLHand' in frame_dict and pid in frame_dict['subjectsWithValidLHand']:
                    left_hand_dict = {'landmarks': person_data['left_hand']['landmarks'], '2D': {}}
                    person_dict['left_hand'] = left_hand_dict
                if 'subjectsWithValidRHand' in frame_dict and pid in frame_dict['subjectsWithValidRHand']:
                    right_hand_dict = {'landmarks': person_data['right_hand']['landmarks'], '2D': {}}
                    person_dict['right_hand'] = right_hand_dict
                map_id[pid] = person_dict

            for panelIdx, camIdx in frame_dict['camIdxArray']:
                person2df = os.path.join(root, 'annot_{}_2d'.format(hv), seqName, frame_str, 'Recon2D_00_{0:02d}_{1}.json'.format(camIdx, frame_str))
                with open(person2df) as f:
                    person2d = json.load(f)

                for person_data in person2d:
                    pid = person_data['id']
                    if pid == -1:
                        continue
                    person_dict = map_id[pid]
                    person_dict['body']['2D'][camIdx] = {'insideImg': person_data['body']['insideImg'], 'occluded': person_data['body']['occluded']}

                    if 'left_hand' in person_dict:
                        person_dict['left_hand']['2D'][camIdx] = {'insideImg': person_data['left_hand']['insideImg'], 'occluded': person_data['left_hand']['self_occluded'],
                                                                  'overlap': person_data['left_hand']['overlap']}

                    if 'right_hand' in person_dict:
                        person_dict['right_hand']['2D'][camIdx] = {'insideImg': person_data['right_hand']['insideImg'], 'occluded': person_data['right_hand']['self_occluded'],
                                                                   'overlap': person_data['right_hand']['overlap']}

            for _, value in map_id.items():
                if seqName == '171204_pose5' or seqName == '171204_pose6':
                    testing_data.append(value)
                else:
                    training_data.append(value)

    with open('./a4_collected.pkl', 'wb') as f:
        pickle.dump({'training_data': training_data, 'testing_data': testing_data}, f)

# collect camera calibration data
if os.path.isfile('./camera_data_a4.pkl'):
    print('Camere file exists.')
else:
    seqs = df.keys()
    calib_dict = {}
    for seqName in seqs:
        cam_dict = {}
        for camIdx in range(31):
            annot_dir = os.path.join(root, 'annot_calib', seqName)
            calib_file = os.path.join(annot_dir, 'calib_00_{:02d}.json'.format(camIdx))
            calib = load_calib_file(calib_file)
            cam_dict[camIdx] = calib
        calib_dict[seqName] = cam_dict
    with open('./camera_data_a4.pkl', 'wb') as f:
        pickle.dump(calib_dict, f)


"""
#################################################################
Panoptic A5
#################################################################
"""
# run in Python 3

root = '/media/posefs0c/panopticdb/a5/'
sample_list = os.path.join(root, 'sample_list.pkl')

with open(sample_list, 'rb') as f:
    df = pickle.load(f)

# collect hand data
if os.path.isfile('./a5_collected.pkl'):
    print('A5 collection file exists.')
else:
    training_data = []
    testing_data = []
    for seqName, seq_samples in df.items():
        i = 0
        for hvframe, frame_dict in seq_samples.items():
            i += 1
            hv, frame_str = hvframe
            print('collecting data: {} {}/{}'.format(seqName, i, len(seq_samples)))
            person3df = os.path.join(root, 'annot_{}_3d'.format(hv), seqName, 'Recon3D_{0}{1}.json'.format(hv, frame_str))
            with open(person3df) as f:
                print(person3df)
                person3d = json.load(f)

            map_id = {}
            for person_data in person3d:
                pid = person_data['id']
                if pid == -1:
                    continue
                person_dict = {'seqName': seqName, 'frame_str': frame_str, 'id': pid}
                body_dict = {'landmarks': person_data['body']['landmarks'], '2D': {}}
                person_dict['body'] = body_dict
                if 'subjectsWithValidLHand' in frame_dict and pid in frame_dict['subjectsWithValidLHand']:
                    left_hand_dict = {'landmarks': person_data['left_hand']['landmarks'], '2D': {}}
                    person_dict['left_hand'] = left_hand_dict
                if 'subjectsWithValidRHand' in frame_dict and pid in frame_dict['subjectsWithValidRHand']:
                    right_hand_dict = {'landmarks': person_data['right_hand']['landmarks'], '2D': {}}
                    person_dict['right_hand'] = right_hand_dict
                map_id[pid] = person_dict

            for panelIdx, camIdx in frame_dict['camIdxArray']:
                person2df = os.path.join(root, 'annot_{}_2d'.format(hv), seqName, frame_str, 'Recon2D_00_{0:02d}_{1}.json'.format(camIdx, frame_str))
                with open(person2df) as f:
                    person2d = json.load(f)

                for person_data in person2d:
                    pid = person_data['id']
                    if pid == -1:
                        continue
                    person_dict = map_id[pid]
                    person_dict['body']['2D'][camIdx] = {'insideImg': person_data['body']['insideImg'], 'occluded': person_data['body']['occluded']}

                    if 'left_hand' in person_dict:
                        person_dict['left_hand']['2D'][camIdx] = {'insideImg': person_data['left_hand']['insideImg'], 'occluded': person_data['left_hand']['self_occluded'],
                                                                  'overlap': person_data['left_hand']['overlap']}

                    if 'right_hand' in person_dict:
                        person_dict['right_hand']['2D'][camIdx] = {'insideImg': person_data['right_hand']['insideImg'], 'occluded': person_data['right_hand']['self_occluded'],
                                                                   'overlap': person_data['right_hand']['overlap']}

            for _, value in map_id.items():
                training_data.append(value)

    with open('./a5_collected.pkl', 'wb') as f:
        pickle.dump({'training_data': training_data, 'testing_data': testing_data}, f)

# collect camera calibration data
if os.path.isfile('./camera_data_a5.pkl'):
    print('Camere file exists.')
else:
    seqs = df.keys()
    calib_dict = {}
    for seqName in seqs:
        cam_dict = {}
        for camIdx in range(31):
            annot_dir = os.path.join(root, 'annot_calib', seqName)
            calib_file = os.path.join(annot_dir, 'calib_00_{:02d}.json'.format(camIdx))
            calib = load_calib_file(calib_file)
            cam_dict[camIdx] = calib
        calib_dict[seqName] = cam_dict
    with open('./camera_data_a5.pkl', 'wb') as f:
        pickle.dump(calib_dict, f)


"""
#################################################################
Panoptic A4Plus
#################################################################
"""
# run in Python 3

root = '/media/posefs0c/panopticdb/a4/'
sample_list = os.path.join(root, 'sample_list.pkl')

with open(sample_list, 'rb') as f:
    df = pickle.load(f)

# collect hand data
if os.path.isfile('./a4plus_collected.pkl'):
    print('A4 collection file exists.')
else:
    training_data = []
    testing_data = []
    for seqName, seq_samples in df.items():
        i = 0
        for hvframe, frame_dict in seq_samples.items():
            i += 1
            hv, frame_str = hvframe
            print('collecting data: {} {}/{}'.format(seqName, i, len(seq_samples)))
            person3df = os.path.join(root, 'annot_{}_3d'.format(hv), seqName, 'Recon3D_{0}{1}.json'.format(hv, frame_str))
            with open(person3df) as f:
                print(person3df)
                person3d = json.load(f)

            map_id = {}
            for person_data in person3d:
                pid = person_data['id']
                if pid == -1:
                    continue
                person_dict = {'seqName': seqName, 'frame_str': frame_str, 'id': pid}
                body_dict = {'landmarks': person_data['body']['landmarks'], '2D': {}}
                person_dict['body'] = body_dict
                if 'subjectsWithValidLHand' in frame_dict and pid in frame_dict['subjectsWithValidLHand']:
                    left_hand_dict = {'landmarks': person_data['left_hand']['landmarks'], '2D': {}}
                    person_dict['left_hand'] = left_hand_dict
                if 'subjectsWithValidRHand' in frame_dict and pid in frame_dict['subjectsWithValidRHand']:
                    right_hand_dict = {'landmarks': person_data['right_hand']['landmarks'], '2D': {}}
                    person_dict['right_hand'] = right_hand_dict
                map_id[pid] = person_dict

            for panelIdx, camIdx in frame_dict['camIdxArray']:
                person2df = os.path.join(root, 'annot_{}_2d'.format(hv), seqName, frame_str, 'Recon2D_00_{0:02d}_{1}.json'.format(camIdx, frame_str))
                with open(person2df) as f:
                    person2d = json.load(f)

                for person_data in person2d:
                    pid = person_data['id']
                    if pid == -1:
                        continue
                    person_dict = map_id[pid]
                    person_dict['body']['2D'][camIdx] = {'insideImg': person_data['body']['insideImg'], 'occluded': person_data['body']['occluded']}

                    if 'left_hand' in person_dict:
                        person_dict['left_hand']['2D'][camIdx] = {'insideImg': person_data['left_hand']['insideImg'], 'occluded': person_data['left_hand']['self_occluded'],
                                                                  'overlap': person_data['left_hand']['overlap']}

                    if 'right_hand' in person_dict:
                        person_dict['right_hand']['2D'][camIdx] = {'insideImg': person_data['right_hand']['insideImg'], 'occluded': person_data['right_hand']['self_occluded'],
                                                                   'overlap': person_data['right_hand']['overlap']}

            for _, value in map_id.items():
                if seqName == '171204_pose5' or seqName == '171204_pose6':
                    testing_data.append(value)
                else:
                    training_data.append(value)

    with open('./a4plus_collected.pkl', 'wb') as f:
        pickle.dump({'training_data': training_data, 'testing_data': testing_data}, f)
