from pathlib import Path
import re
import torch

run6 = Path('run6-frames')
files = list(run6.glob('*.frame'))
eis = []
for file in files:
    m = torch.load(file)
    try:
        batch = int(re.findall(r'\d+', str(file))[1])
        tot_ei = m['ei_layer1'] + m['ei_layer2'] + m['ei_layer3']
        eis.append((batch, tot_ei, m['ei_layer1'], m['ei_layer2'], m['ei_layer3']))
    except:
        pass
eis = sorted(eis, key=lambda k: k[0])
for batch, tot_ei, ei1, ei2, ei3 in eis:
    print(batch, tot_ei, ei1, ei2, ei3)
