from midi2audio import FluidSynth

import os
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser(
    description='''
        make midi file to wav file
    '''
)
parser.add_argument(
    '-input_dir', '--input_dir',
    required=True, type=str, help='midi file directory'
)
parser.add_argument(
    '-output_dir', '--output_dir',
    required=True, type=str, help='output directory for wav files'
)

args = parser.parse_args()

# using the default sound font in 44100 Hz sample rate
fs = FluidSynth()

midi_files = sorted( glob(os.path.join(args.input_dir, '*')) )
output_path = os.path.join(args.output_dir)
os.makedirs(output_path, exist_ok=True)

for file in midi_files:
    f_name = file.split("/")[-1].split('.')[0]+".wav"
    output_file = os.path.join(output_path, f_name)
    fs.midi_to_audio(file, output_file)

print("Done!")
