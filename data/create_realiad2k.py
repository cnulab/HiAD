import os
import json
from tqdm import tqdm
import shutil



if __name__ == '__main__':
    src_root = 'realiad_raw'
    dst_root = 'RealIAD-2K'
    categories = ['bottle_cap', 'mint', 'usb_adaptor']

    for category in categories:
        train_json_path = os.path.join(dst_root, category, 'train.jsonl')
        with open(train_json_path, 'r+') as f:
            train_samples = [ json.loads(line) for line in f.readlines()]

        test_json_path = os.path.join(dst_root, category, 'test.jsonl')
        with open(test_json_path, 'r+') as f:
            test_samples = [json.loads(line) for line in f.readlines()]

        samples = train_samples + test_samples

        for sample in tqdm(samples, total=len(samples), desc='Extraction {}'.format(category)):
            dst_file_path = sample['filename']

            file_name = os.path.basename(dst_file_path)
            if file_name.find('NG') != -1:
                realiad_sample_name = "S{}".format(file_name.split('_')[-5])
                readiad_label_name = file_name.split('_')[-4]
                readiad_anomaly_name = file_name.split('_')[-3]
                src_file_path = os.path.join(src_root, category, readiad_label_name,
                                             readiad_anomaly_name, realiad_sample_name, file_name)
            else:
                realiad_sample_name = "S{}".format(file_name.split('_')[-4])
                readiad_label_name = file_name.split('_')[-3]
                src_file_path = os.path.join(src_root, category, readiad_label_name,
                                              realiad_sample_name, file_name)

            dst_file_root = os.path.join(dst_root, os.path.dirname(dst_file_path))
            os.makedirs(dst_file_root, exist_ok=True)
            shutil.copy(src_file_path, os.path.join(dst_file_root, file_name))

            if 'mask' in sample:
                dst_mask_path = sample['mask']
                mask_name = os.path.basename(dst_mask_path)
                if mask_name.find('NG') != -1:
                    realiad_sample_name = "S{}".format(mask_name.split('_')[-5])
                    readiad_label_name = mask_name.split('_')[-4]
                    readiad_anomaly_name = mask_name.split('_')[-3]
                    src_mask_path = os.path.join(src_root, category, readiad_label_name,
                                                 readiad_anomaly_name, realiad_sample_name, mask_name)
                else:
                    realiad_sample_name = "S{}".format(mask_name.split('_')[-4])
                    readiad_label_name = mask_name.split('_')[-3]
                    src_mask_path = os.path.join(src_root, category, readiad_label_name,
                                                 realiad_sample_name, mask_name)

                dst_mask_root = os.path.join(dst_root, os.path.dirname(dst_mask_path))
                os.makedirs(dst_mask_root, exist_ok=True)
                shutil.copy(src_mask_path, os.path.join(dst_mask_root, mask_name))

    print('Checking files...')
    train_json_path = os.path.join(dst_root, 'train_uni.jsonl')
    with open(train_json_path, 'r+') as f:
        train_samples = [ json.loads(line) for line in f.readlines()]

    test_json_path = os.path.join(dst_root, 'test_uni.jsonl')
    with open(test_json_path, 'r+') as f:
        test_samples = [json.loads(line) for line in f.readlines()]

    samples = train_samples + test_samples
    for sample in samples:
        assert os.path.exists(os.path.join(dst_root,sample['filename']))
        if 'mask' in sample:
            assert os.path.exists(os.path.join(dst_root, sample['mask']))

    print('RealIAD-2K dataset created successfully. Located at: {}.'.format(dst_root))



