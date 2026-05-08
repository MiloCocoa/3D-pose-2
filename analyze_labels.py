import json
import glob
import os

data_dir = 'C:/Users/User/Desktop/GitHub/3D-pose-2/data/test-pos-seq-20260502/'
json_files = glob.glob(os.path.join(data_dir, '**/*.json'), recursive=True)

descent_true = 0
descent_false = 0
ascent_true = 0
ascent_false = 0

for filepath in json_files:
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            labels = data.get('metadata', {}).get('label', [])
            if len(labels) >= 10:
                descent = str(labels[7]).lower()
                ascent = str(labels[9]).lower()
                
                if descent == 'true':
                    descent_true += 1
                elif descent == 'false':
                    descent_false += 1
                    
                if ascent == 'true':
                    ascent_true += 1
                elif ascent == 'false':
                    ascent_false += 1
        except Exception as e:
            print(f'Error reading {filepath}: {e}')

print(f'Descent - True: {descent_true}, False: {descent_false}')
print(f'Ascent  - True: {ascent_true}, False: {ascent_false}')
print(f'Total files checked: {len(json_files)}')
