import json
import os

def make_data_list(wavscp_path, transcript_path, save_path):
    txt_dict = {}
    with open(wavscp_path, 'r', encoding='utf-8') as scpreader:
        line = scpreader.readline()
        while(line):
            info = line.strip().split(' ')
            assert len(info) == 2
            txt_dict[info[0]] = info[1]
            line = scpreader.readline()
    scpreader.close()

    with open(transcript_path, 'r', encoding='utf-8') as transcriptreader, \
        open(os.path.join(save_path, 'data.list'), 'w', encoding='utf-8') as writer:
        line = transcriptreader.readline()
        while(line):
            write_line = {}
            info = line.strip().split(' ', maxsplit=1)
            assert len(info) == 2
            write_line['key'] = info[0]
            write_line['wav'] = txt_dict[info[0]]
            write_line['txt'] = info[1]
            write_line = json.dumps(write_line, ensure_ascii=False)
            writer.write(write_line + '\n')
            line = transcriptreader.readline()
    transcriptreader.close()
    writer.close()

if __name__ == "__main__":
    info_path = './data'
    save_path = './raw'
    os.makedirs(save_path, exist_ok=True)
    sets = ['test', 'dev', 'train_nodup']
    for single_set in sets:
        pre_path = os.path.join(info_path, single_set)
        sub_path = os.path.join(save_path, single_set)
        os.makedirs(sub_path, exist_ok=True)
        make_data_list(os.path.join(pre_path, 'wav.scp'), os.path.join(pre_path, 'text'), sub_path)
    print('done')