from utils.utils import oneline_wrapper
import os
import time
from multiprocessing import Process
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo",       type=str, default='maxp_56_6')
    parser.add_argument("-seed",    type=int,            default=6)
    parser.add_argument("-model",       type=str,               default="MaxPressure")
    parser.add_argument("-eightphase", action="store_true",     default=True)
    parser.add_argument("-multi_process", action="store_true",  default=False)
    parser.add_argument("-workers",     type=int,               default=1)
    parser.add_argument("-r34", action="store_true", default=False)
    parser.add_argument("-r54", action="store_true", default=False)
    parser.add_argument("-r56", action="store_true", default=True)
    parser.add_argument("-r45", action="store_true", default=False)
    parser.add_argument("-hangzhou",    action="store_true",    default=False)
    parser.add_argument("-jinan",       action="store_true",    default=False)
    parser.add_argument("-newyork", action="store_true", default=False)
    return parser.parse_args()


def main(in_args):

    if in_args.hangzhou:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json",
                             "anon_4_4_hangzhou_real_5816.json"]
        template = "Hangzhou"
    elif in_args.jinan:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json"]
        template = "Jinan"
    elif in_args.newyork:
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        template = "newyork_28_7"
    elif in_args.r34:
        num_rounds = 80
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["flow_3_4.json"]
        template = "3_4"
    elif in_args.r45:
        num_rounds = 80
        count = 3600
        road_net = "4_5"
        traffic_file_list = ["flow_4_5.json"]
        template = "4_5"
    elif in_args.r54:
        num_rounds = 80
        count = 3600
        road_net = "5_4"
        traffic_file_list = ["flow_5_4.json"]
        template = "5_4"
    elif in_args.r56:
        num_rounds = 80
        count = 3600
        road_net = "5_6"
        traffic_file_list = ["flow_5_6.json"]
        template = "5_6"
    else:
        count = 3600
        road_net = "33_34"
        traffic_file_list = ["flow_33_34.json"]
        template = "33_34"
    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)
    process_list = []

    for traffic_file in traffic_file_list:
        dic_traffic_env_conf_extra = {
            "SEED": in_args.seed,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,

            "MODEL_NAME": in_args.model,
            "RUN_COUNTS": count,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue",
            ],

            "DIC_REWARD_INFO": {
                "pressure": 0
            },
        }

        if in_args.eightphase:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            }
            dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                        'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

        dic_agent_conf_extra = {
            "FIXED_TIME": [15, 15, 15, 15],
        }
        dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_" +
                                          time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_" +
                                                   time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net))
        }

        if in_args.multi_process:
            process_list.append(Process(target=oneline_wrapper,
                                        args=(dic_agent_conf_extra,
                                              dic_traffic_env_conf_extra, dic_path_extra))
                                )
        else:
            oneline_wrapper(dic_agent_conf_extra, dic_traffic_env_conf_extra, dic_path_extra)

    if in_args.multi_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < in_args.workers:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < in_args.workers:
                continue

        for p in list_cur_p:
            p.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
