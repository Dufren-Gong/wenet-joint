# Copyright (c) 2020 Mobvoi Inc (Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import sys

import json
import torch
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--evaluation_mode',
                        default='acc')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    mode_arr_name = args.evaluation_mode + '_arr'
    with open(args.src_path, 'r', encoding='utf-8') as raader:
        infos = raader.readline()
        while(infos):
            infos = json.loads(infos.strip())
            if mode_arr_name in infos.keys():
                break
            infos = raader.readline()
    raader.close()
    scores = infos[mode_arr_name]
    assert len(scores) >= args.num
    if args.evaluation_mode == 'wer':
        reverse_mode = False
    else:
        reverse_mode = True
    sorted_val_scores = sorted(scores,
                               key=lambda x: x[1],
                               reverse=reverse_mode)[:args.num]
    print(sorted_val_scores)
    dir = os.path.dirname(args.src_path)
    path_list = [os.path.join(dir, 'epoch_' + str(i[0]) + '.pt') for i in sorted_val_scores]
    avg = {}
    for path in tqdm(path_list):
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        for k in states.keys():
            if k not in avg.keys():
                avg[k] = states[k].clone()
            else:
                avg[k] += states[k]
    # average
    for k in tqdm(avg.keys()):
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], args.num)
    dst_model = os.path.join(dir, args.evaluation_mode + '_avg_' + str(args.num) + '.pt')
    print('Saving to {}'.format(dst_model))
    torch.save(avg, dst_model)

if __name__ == '__main__':
    main()
