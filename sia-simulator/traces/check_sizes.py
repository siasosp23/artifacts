import pandas as pd
import os

num_rows_to_configs = {}
models = os.listdir('./')
models.remove('philly.csv')
models.remove('check_sizes.py')
for model in models:
    for device in ['rtx', 'dgx', 'aws']:
        df = pd.read_csv('%s/placements-%s.csv'%(model, device))
        bszs = set(df[['local_bsz']].to_numpy()[:, 0])
        for bsz in bszs:
            if df[df['local_bsz']==bsz].shape[0] in num_rows_to_configs:
                num_rows_to_configs[df[df['local_bsz']==bsz].shape[0]].append((model, device, bsz))
            else:
                num_rows_to_configs[df[df['local_bsz']==bsz].shape[0]] = [(model, device, bsz)]
