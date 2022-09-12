from collections import Counter
import pickle
import glob
import utils
from tqdm import tqdm

def extract_events(input_path, chord=False):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    if chord:
        chord_items = utils.extract_chords(note_items)
        items = chord_items + tempo_items + note_items
    else:
        items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events

all_elements= []
for midi_file in tqdm(glob.glob("/home/bang/다운로드/REMI_dataset/data/*/*.mid*", recursive=True)):
    events = extract_events(midi_file, chord=True) # If you're analyzing chords, use `extract_events(midi_file, chord=True)`
    for event in events:
        element = '{}_{}'.format(event.name, event.value)
        all_elements.append(element)

counts = Counter(all_elements)
event2word = {c: i for i, c in enumerate(counts.keys())}
word2event = {i: c for i, c in enumerate(counts.keys())}

# print(len(event2word))

pickle.dump((event2word, word2event), open('dictionary.pkl', 'wb'))