from threading import Thread
import json
from tqdm import tqdm
import torchaudio
import numpy as np
import os
import random

torchaudio.utils.sox_utils.set_buffer_size(16500)

def check_nan_inf(np_arr):
    # 检查数组是否包含NaN或Inf
    has_nan = np.isnan(np_arr).any()
    has_inf = np.isinf(np_arr).any()
    return (has_nan or has_inf)

def read_json_files(path):
    with open(path, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()[1616765:]
    reader.close()
    random.shuffle(lines)
    lines = [json.loads(line.strip()) for line in lines]
    return lines

def split_sub(np_arr, split_num):
    temp_arr = []
    gap = int(len(np_arr) / split_num)
    for i in range(split_num):
        if i != split_num - 1:
            temp_arr.append(np_arr[i * gap:(i + 1) * gap])
        else:
            temp_arr.append(np_arr[i * gap:])
    return temp_arr

def one_thread(path, dict_arr):
    error_dict_arr = []
    for line in tqdm(dict_arr):
        wav_file = line['wav']
        if 'start' in line:
            assert 'end' in line
            sample_rate = torchaudio.backend.sox_io_backend.info(
                wav_file).sample_rate
            start_frame = int(float(line['start']) * sample_rate)
            end_frame = int(float(line['end']) * sample_rate)
            #在此处获取到了有标记数据段的语言数据
            waveform, _ = torchaudio.backend.sox_io_backend.load(
                filepath=wav_file,
                num_frames=end_frame - start_frame,
                frame_offset=start_frame)
        else:
            waveform, _ = torchaudio.load(wav_file)
        assert len(waveform) > 0
        if check_nan_inf(waveform):
            error_dict_arr.append(line)

    with open(path, 'a', encoding='utf-8') as writer:
        for line in error_dict_arr:
            writer.write(line)
    writer.close()

def main():
    print(f'pwd : {os.getcwd()}')
    data_list_path = './data/total_train/data.list'
    lines = read_json_files(data_list_path)
    lines_arr = split_sub(lines, 32)
    error_path ='./error'
    #构成多线程数组
    each_threads = [Thread(target=one_thread, args=(error_path, line_arr)) for line_arr in lines_arr]
    for thread in each_threads:
        thread.start()
    #等待最后一个线程结束
    each_threads[-1].join()

if __name__ == "__main__":
    main()