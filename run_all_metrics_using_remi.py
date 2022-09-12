import os, sys
import warnings
from itsdangerous import exc
warnings.simplefilter(action='ignore', category=FutureWarning)

import miditoolkit
import utils

import itertools, random
import scipy.stats
from scipy.io import loadmat

import numpy as np
import pandas as pd
from glob import glob
from argparse import ArgumentParser

from tqdm import tqdm


N_PITCH_CLS = 12 # (C, C#, ..., B)

# support function
def get_event_seq(midi_path):

    # read midi inti "Item"
    note_items, tempo_items = utils.read_items(midi_path)
    # Quantize note items
    note_items = utils.quantize_items(note_items)
    # extract chord
    chord_items = utils.extract_chords(note_items)
    # group items
    items = chord_items + tempo_items + note_items
    max_time = note_items[-1].end
    groups = utils.group_items(items, max_time)
    # "Item" to "Event"
    events = utils.item2event(groups)

    return events

def bar_check(events):
    bar_markers = []
    for ind, event in enumerate(events):
        if event.name == 'Bar':
            bar_markers.append(ind)

    return bar_markers

def get_bars_crop(ev_seq, start_bar, end_bar, verbose=False):
    
    if start_bar < 0 or end_bar < 0:
        raise ValueError('Invalid start_bar: {}, or end_bar: {}.'.format(start_bar, end_bar))

    # get the indices of ``Bar`` events
    bar_markers = bar_check(ev_seq)

    if start_bar > len(bar_markers) - 1:
        raise ValueError('start_bar: {} beyond end of piece.'.format(start_bar))

    if end_bar < len(bar_markers) - 1:
        cropped_seq = ev_seq[ bar_markers[start_bar] : bar_markers[end_bar + 1] ]
    else:
        if verbose:
            print (
            '[Info] end_bar: {} beyond or equal the end of the input piece; only the last {} bars are returned.'.format(
                end_bar, len(bar_markers) - start_bar
            ))
    cropped_seq = ev_seq[ bar_markers[start_bar] : ]

    return cropped_seq

def compute_histogram_entropy(hist):
    return scipy.stats.entropy(hist) / np.log(2)

def get_pitch_histogram(ev_seq, verbose=False):
    
    ev_seq = [x.value for x in ev_seq if x.name == 'Note On']

    if not len(ev_seq):
        if verbose:
            print ('[Info] The sequence contains no notes.')
        return None

    # compress sequence to pitch classes & get normalised counts
    ev_seq = pd.Series(ev_seq) % N_PITCH_CLS
    ev_hist = ev_seq.value_counts(normalize=True)

    # make the final histogram
    hist = np.zeros( (N_PITCH_CLS,) )
    for i in range(N_PITCH_CLS):
        if i in ev_hist.index:
            hist[i] = ev_hist.loc[i]

    return hist

def get_octave_histogram(ev_seq, verbose=False):
    
    ev_seq = [x.value for x in ev_seq if x.name == 'Note On']

    if not len(ev_seq):
        if verbose:
            print ('[Info] The sequence contains no notes.')
        return None

    # compress sequence to pitch classes & get normalised counts
    ev_seq = pd.Series(ev_seq) // N_PITCH_CLS
    ev_hist = ev_seq.value_counts(normalize=True)

    # make the final histogram
    hist = np.zeros( (11,) )
    for i in range(11):
        if i in ev_hist.index:
            hist[i] = ev_hist.loc[i]

    return hist

def get_onset_xor_distance(seq_a, seq_b, pos_evs):
    # sanity checks
    # assert seq_a[0].name == 'Bar' and seq_b[0].name == 'Bar'
    # assert len(bar_check(seq_a)) == 1 and len(bar_check(seq_b)) == 1

    # compute binary onset vectors
    n_pos = pos_evs
    def make_onset_vec(seq):
        cur_pos = -1
        onset_vec = np.zeros((n_pos,))
        for ev in seq:
            if ev.name == 'Position':
                cur_pos = int(ev.value.split('/')[0])-1
            if ev.name in 'Note On':
                onset_vec[cur_pos] = 1
        return onset_vec
    a_onsets, b_onsets = make_onset_vec(seq_a), make_onset_vec(seq_b)

    # compute XOR distance
    dist = np.sum( np.abs(a_onsets - b_onsets) ) / n_pos
    return dist

def read_fitness_mat(fitness_mat_file):
    ext = os.path.splitext(fitness_mat_file)[-1].lower()

    if ext == '.npy':
        f_mat = np.load(fitness_mat_file)
    elif ext == '.mat':
        mat_dict = loadmat(fitness_mat_file)
        f_mat = mat_dict['fitness_info'][0, 0][0]
        f_mat[ np.isnan(f_mat) ] = 0.0
    else:
        raise ValueError('Unsupported fitness scape plot format: {}'.format(ext))

    for slen in range(f_mat.shape[0]):
        f_mat[slen] = np.roll(f_mat[slen], slen // 2)

    return f_mat

# metric 계산 함수들
def compute_piece_pitch_entropy(piece_ev_seq, window_size, verbose = False):
    n_bars = len(bar_check(piece_ev_seq))

    if window_size > n_bars:
        print ('[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.'.format(window_size))
        window_size = n_bars
    

    # compute entropy of all possible segments
    pitch_ents = []
    for st_bar in range(0, n_bars - window_size + 1):
        seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1)
        pitch_hist = get_pitch_histogram(seg_ev_seq)
        if pitch_hist is None:
            if verbose:
                print ('[Info] No notes in this crop: {}~{} bars.'.format(st_bar, st_bar + window_size - 1))
            continue

        pitch_ents.append( compute_histogram_entropy(pitch_hist) )

    return np.mean(pitch_ents)


def compute_piece_octave_entropy(piece_ev_seq, window_size, verbose = False):
    n_bars = len(bar_check(piece_ev_seq))

    if window_size > n_bars:
        print ('[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.'.format(window_size))
        window_size = n_bars
    

    # compute entropy of all possible segments
    octave_ents = []
    for st_bar in range(0, n_bars - window_size + 1):
        seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1)
        octave_hist = get_octave_histogram(seg_ev_seq)
        if octave_hist is None:
            if verbose:
                print ('[Info] No notes in this crop: {}~{} bars.'.format(st_bar, st_bar + window_size - 1))
            continue

        octave_ents.append( compute_histogram_entropy(octave_hist) )

    return np.mean(octave_ents)

def compute_piece_groove_similarity(piece_ev_seq, pos_evs=16, max_pairs=1000):
    # remove redundant ``Bar`` marker
    if piece_ev_seq[-1].name == 'Bar':
        piece_ev_seq = piece_ev_seq[:-1]

    # get every single bar & compute indices of bar pairs
    n_bars = len(bar_check(piece_ev_seq))
    bar_seqs = []
    for b in range(n_bars):
        bar_seqs.append( get_bars_crop(piece_ev_seq, b, b) )
    pairs = list( itertools.combinations(range(n_bars), 2) )
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    # compute pairwise grooving similarities
    grv_sims = []
    for p in pairs:
        grv_sims.append(
            1. - get_onset_xor_distance(bar_seqs[p[0]], bar_seqs[p[1]], pos_evs)
        )

    return np.mean(grv_sims)

def compute_piece_chord_progression_irregularity(remi_ev_seq, ngram=3):
    chord_seq = [ev.value for ev in remi_ev_seq if ev.name == 'Chord']
    if len(chord_seq) <= ngram:
        return 1.

    num_ngrams = len(chord_seq) - ngram
    unique_set = set()
    for i in range(num_ngrams):
        str_repr = '_'.join(['-'.join(str(x)) for x in chord_seq[i : i + ngram]])
        if str_repr not in unique_set:
            unique_set.add(str_repr)

    return len(unique_set) / num_ngrams

def compute_structure_indicator(mat_file, low_bound_sec=0, upp_bound_sec=128, sample_rate=2):
    assert low_bound_sec > 0 and upp_bound_sec > 0, '`low_bound_sec` and `upp_bound_sec` should be positive, got: low_bound_sec={}, upp_bound_sec={}.'.format(low_bound_sec, upp_bound_sec)
    low_bound_ts = int(low_bound_sec * sample_rate) - 1
    upp_bound_ts = int(upp_bound_sec * sample_rate)
    f_mat = read_fitness_mat(mat_file)

    if low_bound_ts >= f_mat.shape[0]:
        score = 0
    else:
        score = np.max(f_mat[ low_bound_ts : upp_bound_ts ])

    return score

# 결과 출력 함수
def write_report(result_dict, out_csv_file):
    df = pd.DataFrame().from_dict(result_dict)
    df = df.append(df.agg(['mean']))
    df = df.round(4)
    df.loc['mean', 'piece_name'] = 'DATASET_MEAN'
    df.to_csv(out_csv_file, index=False, encoding='utf-8')

if __name__ == '__main__':
    parser = ArgumentParser(
        description='''
            Runs all evaluation metrics on the pieces within the provided directory, and writes the results to a report.
        '''
    )
    parser.add_argument(
        '-s', '--symbolic_dir',
        required=True, type=str, help='directory containing symbolic musical pieces.'
    )
    parser.add_argument(
        '-p', '--scplot_dir',
        required=True, type=str, help='directory containing fitness scape plots (of the exact SAME pieces as in ``symbolic_dir``).'
    )
    parser.add_argument(
        '-o', '--out_csv',
        required=True, type=str, help='path to output file for results.'  
    )
    parser.add_argument(
        '--timescale_bounds',
        nargs='+', type=int, default=[3, 8, 15], help='timescale bounds (in secs, [short, mid, long]) for structureness indicators.'
    )
    args = parser.parse_args()

    test_pieces = sorted( glob(os.path.join(args.symbolic_dir, '*')) )
    test_pieces_scplot = sorted( glob(os.path.join(args.scplot_dir, '*')) )

    # print (test_pieces, test_pieces_scplot)
    
    result_dict = {
        'piece_name': [],
        'H1': [],
        'H4': [],
        'GS': [],
        'CPI': [],
        'SI_short': [],
        'SI_mid': [],
        'SI_long': []    
    }

    # plot수랑 test_midi수가 같아야함!
    assert len(test_pieces) == len(test_pieces_scplot), 'detected discrepancies between 2 input directories.'

    for p, p_sc in tqdm(zip(test_pieces, test_pieces_scplot)):
    # for p in tqdm(test_pieces):
        # print ('>> now processing: {}'.format(p))
        try:
            seq = get_event_seq(p)
        except:
            print(p)
            continue
        result_dict['piece_name'].append(p.replace('\\', '/').split('/')[-1])
        h1 = compute_piece_pitch_entropy(seq, 1)
        # h1 = compute_piece_octave_entropy(seq, 1)
        result_dict['H1'].append(h1)
        h4 = compute_piece_pitch_entropy(seq, 4)
        # h4 = compute_piece_octave_entropy(seq, 4)
        result_dict['H4'].append(h4)
        gs = compute_piece_groove_similarity(seq, pos_evs=64)
        result_dict['GS'].append(gs)
        cpi = compute_piece_chord_progression_irregularity(seq)
        result_dict['CPI'].append(cpi)
        si_short = compute_structure_indicator(p_sc, args.timescale_bounds[0], args.timescale_bounds[1])
        result_dict['SI_short'].append(si_short)
        si_mid = compute_structure_indicator(p_sc, args.timescale_bounds[1], args.timescale_bounds[2])
        result_dict['SI_mid'].append(si_mid)
        si_long = compute_structure_indicator(p_sc, args.timescale_bounds[2])
        result_dict['SI_long'].append(si_long)

        print ('  1-bar H: {:.3f}'.format(h1))
        print ('  4-bar H: {:.3f}'.format(h4))
        print ('  GS: {:.4f}'.format(gs))
        print ('  CPI: {:.4f}'.format(cpi))
        print ('  SI_short: {:.4f}'.format(si_short))
        print ('  SI_mid: {:.4f}'.format(si_mid))
        print ('  SI_long: {:.4f}'.format(si_long))
        print ('==========================')

    if len(result_dict):
        write_report(result_dict, args.out_csv)
    else:
        print ('No pieces are found !!')