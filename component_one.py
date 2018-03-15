import numpy as np

def turn_to_dict(file_name):
    beacon_dict = {}

    with open(file_name, 'r') as packs:
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


def train(state_files, num_beacon):
    # Currently fake data

    num_state = len(state_files)
    state_map = np.zeros((num_state, num_beacon, 2))

    for file, si in zip(state_files, range(num_state)):
        beacon_dict = turn_to_dict(file)
        for rssis in beacon_dict:
            arr = beacon_dict[rssis]
            state_map[si][int(rssis[1:])-1][0] = np.mean(arr)
            state_map[si][int(rssis[1:])-1][1] = np.var(arr)

    return state_map


def process_test(test_name, num_beacon):
    # Turn original file to numpy array with [avg_b1_rssi, avg_b2_rssi, ..., avg_bn_rssi]

    beacon_dict = turn_to_dict(test_name)
    avg = np.zeros(num_beacon)

    for bi in beacon_dict:
        index = int(bi[1:])
        avg[index - 1] = np.mean(beacon_dict[bi])

    return avg


def test(state_map, test_file):
    # return state index with max liklihood
    # state_map => (state, beacon, (mu, sigma))
    # test_rssi => [avg_rssi]

    test_rssi = process_test(test_file, 2)
    maxm_sum_log = 0.0
    maxm_index = -1

    for i in range(state_map.shape[0]):
        map_mu = state_map[i][:, 0]
        map_sigma = state_map[i][:, 1]
        sum_log = - np.sum(np.square(test_rssi - map_mu) / (2 * map_sigma ** 2)) - \
                  np.sum(np.log(map_sigma))
        if maxm_index == -1 or maxm_sum_log < sum_log:
            maxm_index, maxm_sum_log = i, sum_log

    return maxm_index

def main():
    state_files = ['s1', 's2']
    state_map = train(state_files, 2)
    res1 = test(state_map, 's1')
    res2 = test(state_map, 's2')
    print(res1, res2) # Expecting output of 0 1 in this case.


main()