import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import groundTruthLocation as gtl

freq = -15
intv = 0.1

max_time = 0
train_files = []
for i in range(1, 13):
    train_files.append('horusrssi/' + str(freq) +'dBm' + str(intv) + 'secRssi' + str(i) + '.txt')

test_file = 'traceRssi' + str(freq) +'dB' + str(intv) + 'sec1.txt'

def convert_train_data(file_name):
    # File content should be [b_i, rssi]
    #                        ..........
    # Convert it to dictionary dict[index_beacon] = [rss_1, ..., rssi_n]

    beacon_dict = {}
    with open(file_name, 'r') as packs:
        packs.readline()
        for pack in packs:
            pack = pack[0: len(pack) - 1]
            space_loc = pack.find(' ')
            beacon = pack[0: space_loc]
            ri = float(pack[space_loc + 1:])

            if beacon not in beacon_dict:
                beacon_dict[beacon] = [ri]
            else:
                beacon_dict[beacon].append(ri)

    return beacon_dict

def get_state_location(file_name = 'standingLocation.csv'):
    # Assumption that metric in csv file is inch, multiply by 0.0254 to get meters.

    with open(file_name, 'r') as file:
        file_len = sum(1 for _ in file)

    with open(file_name, 'r') as file:
        state_loc = np.zeros((file_len, 2))
        for line in file:
            line = line[0:(len(line)-1)]
            splt = line.split(',')
            state_loc[int(splt[0])-1][0] = float(splt[1])
            state_loc[int(splt[0])-1][1] = float(splt[2])

    return 0.0254 * state_loc


def convert_test_data(file_name):
    # file header should be [TimeStamp, Beacon, Rssi]
    # convert it to numpy array

    with open(file_name, 'r') as file:
        file_len = sum(1 for _ in file)

    with open(file_name) as file:
        file.readline() # remove the header
        data = np.zeros((file_len - 1, 3))
        for line, i in zip(file, range(file_len - 1)):
            line = line[0:(len(line)-1)]
            splt = line.split(' ')
            data[i][0] = float(splt[0])
            data[i][1] = int(splt[1])
            data[i][2] = float(splt[2])

    return data


def get_alpha(beacon_dict, num_beacon = 60):
    # Returned alphas should be [alpha_b1, alpha_b2, ..., alpha_bn]
    # ssvs => signal strength values

    alphas = np.zeros(num_beacon)

    for beacon in beacon_dict:
        ssvs = np.array(beacon_dict[beacon])
        num_ssv = len(ssvs)

        shift_vec = np.arange(num_ssv - 1) + 1
        ssvs_shift = ssvs[shift_vec]
        ssvs_trunc = ssvs[0: len(ssvs) - 1]

        sbar = np.mean(ssvs)

        sumsq = np.sum((ssvs - sbar) ** 2)
        if sumsq == 0:   # Need discussion with subham
            r1 = 0
        else:
            r1 = abs(np.sum((ssvs_shift - sbar) * (ssvs_trunc - sbar))) / sumsq

        alphas[int(beacon) - 1] = r1

    return alphas

def train(state_files, num_beacon = 60):
    # Get mu and sigma for each state, noted that here it's sigma instead of sigma square.

    num_state = len(state_files)
    state_map = np.zeros((num_state, num_beacon, 2))  # For beacons not in file, exclude them
    state_map[:,:,0] = -100
    state_map[:,:,1] = 1/3

    for file, si in zip(state_files, range(num_state)): # si represents state_index
        beacon_dict = convert_train_data(file)
        alphas = get_alpha(beacon_dict)   # Alphas => [alpha_b1, alpha_b2, ..., alpha_bn]

        for rssis in beacon_dict:  # rssis is beacon number
            alpha = alphas[int(rssis)-1]
            arr = beacon_dict[rssis]
            state_map[si][int(rssis)-1][0] = np.mean(arr)
            state_map[si][int(rssis)-1][1] = np.sqrt((1 + alpha) / (1 - alpha)*np.var(arr))

            if state_map[si][int(rssis)-1][1] == 0:
                state_map[si][int(rssis) - 1][1] = 15
                #state_map[si][int(rssis) - 1][0] = -100  # Subtle!!! Depends on signal strength.

    return state_map

def process_cluster(cluster, num_beacon = 60):
    # Convert cluster to [beacon_1_avg_rssi, ..., beacon_n_avg_rssi], 0 for beacons not present

    ret = np.zeros(num_beacon)
    cnt = np.zeros(num_beacon)

    arg = np.argsort(cluster[:,1])
    cluster = cluster[arg]

    for i in range(len(cluster)):
        ret[int(cluster[i][1])-1] += cluster[i][2]
        cnt[int(cluster[i][1])-1] += 1

    cnt[cnt == 0] = 1.0

    return ret / cnt

def cluster_test_data(filename, num_beacon = 60, interval = 10):
    # Data should be numpy array
    # Return dictionary with d['i'] = [beacon_1_avg_rssi, ..., beacon_n_avg_rssi], i represents state_i

    data = convert_test_data(filename)
    num_rows = data.shape[0]
    global max_time
    max_time = data[num_rows-1][0]

    num_cluster = int(max_time / interval)
    start_index = [0]
    d = {}

    for i in range(1, num_cluster + 1):
        thred = interval * i
        index = np.searchsorted(data[:,0], thred)
        start_index.append(index)

    for i in range(len(start_index) - 1):
        cluster = data[start_index[i]:start_index[i+1]]
        d[i] = process_cluster(cluster)

    return d

def test(state_map, test_rssi, state_loc, index, jud):
    jud = 1 - jud
    # return state index with max liklihood
    # state_map => (state, beacon, (mu, sigma))
    # state_loc => (state, (x, y))
    # test_rssi => [avg_rssi]

    probs = []
    arg_zero = (abs(test_rssi) > 1.0) * jud

    for i in range(state_map.shape[0]):
        map_mu = state_map[i][:, 0]
        map_sigma = state_map[i][:, 1]

        sum_log = - np.sum(arg_zero * np.square(test_rssi - map_mu) / (2 * map_sigma ** 2)) - \
                  np.sum(np.log(map_sigma) * arg_zero)

        probs.append(sum_log)
    probs = np.asarray(probs).reshape(state_map.shape[0], 1)

    len_probs = len(probs)
    norm_probs = np.zeros(len_probs).reshape(state_map.shape[0], 1)

    for i in range(len_probs):
        if np.any(probs - probs[i] > 50):
            norm_probs[i] = 0
        else:
            norm_probs[i] = 1 / (np.sum(np.exp(probs - probs[i])))

    res = np.sum(norm_probs * state_loc, axis = 0)

    return res


def ss_compensator(prev_state, test_file, state_loc, state_map, thred = 2, d = 0.05, N = 6):
    # thred => max distance per second
    # prev_state, pred_state => numpy (x, y) in meter metric
    # d => perturbation fraction

    pred_state = np.array([])
    min_dist = 0
    final_pert = -1
    test_rssi = process_test(test_file, 2)
    max_index = np.argmax(test_rssi)

    for pert in [-0.05, 0, 0.05]:
        pert_arr = np.zeros(len(test_rssi))
        pert_arr[max_index] = pert

        cur_pred = test(state_map, test_rssi + pert_arr, state_loc)
        cur_dist = nl.norm(prev_state - cur_pred)

        if final_pert == -1 or min_dist > cur_dist:
            final_pert = pert
            pred_state = cur_pred
            min_dist = cur_dist

    return pred_state


def main():
    state_loc = get_state_location()
    state_map = train(train_files)

    jud = np.zeros(60)

    for i in range(12):
        cur = state_map[i]
        for j in range(60):
            if cur[j][0] == -100:
                jud[j] = 1

    d = cluster_test_data(test_file)

    pred_loc = np.zeros((len(d), 2))

    for index in d:
        res = test(state_map, d[index], state_loc, index, jud)
        pred_loc[index-1][0] = res[0]
        pred_loc[index-1][1] = res[1]

    global max_time
    trueLoc = np.zeros((int(max_time / 10), 2))
    for i in range(int(max_time / 10)):
        cur = gtl.findActualLocation(startTime=10*(i), endTime=10*(i+1), stopTime=10, maxTime=max_time)
        trueLoc[i][0], trueLoc[i][1] = cur[0], cur[1]

    valid_x = 1 - (trueLoc[:,0] < 4) - (trueLoc[:,0] > 8)
    valid_y = trueLoc[:,1] < 4
    valid_loc = valid_x * valid_y

    errors = valid_loc * np.linalg.norm(pred_loc - trueLoc, axis = 1)
    avg_error = np.sum(errors) / np.sum(valid_loc)

    print(avg_error)

    #errors[errors == 0] = avg_error
    #plt.plot(np.arange(int(max_time / 10)), abs(errors))
    #plt.show()
    #print(pred_loc - trueLoc)

    #for i in range(len(trueLoc) - 1):
    #    print(np.linalg.norm(pred_loc[i] - pred_loc[i+1]))
    #print(valid_loc)

main()
