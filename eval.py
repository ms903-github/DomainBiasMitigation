import utils
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="baseline",
    )
    parser.add_argument(
        "--m_load_path", type=str, default="record/imdb_baseline/baseline_soft/test_male_result.pkl"
    )
    parser.add_argument(
        "--f_load_path", type=str, default="record/imdb_baseline/baseline_soft/test_female_result.pkl"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode
    m_load_path = args.m_load_path
    f_load_path = args.f_load_path

    m_loaded = utils.load_pkl(m_load_path)
    f_loaded = utils.load_pkl(f_load_path)

    if mode == "baseline":
        f_preds, m_preds = f_loaded["predict_labels"], m_loaded["predict_labels"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode)
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}.txt".format(mode)
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

    elif mode == "domain_independent":
        f_preds, m_preds = f_loaded["prediction_conditional"], m_loaded["prediction_conditional"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode + "/prediction_conditional")
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}_{}.txt".format(mode, "conditional")
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

        f_preds, m_preds = f_loaded["prediction_sum_out"], m_loaded["prediction_sum_out"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode + "/prediction_sum_out")
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}_{}.txt".format(mode, "sumout")
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

    elif mode == "domain_discriminative":
        f_preds, m_preds = f_loaded["prediction_sum_prob_wo_prior_shift"], m_loaded["prediction_sum_prob_wo_prior_shift"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode + "/sum_wo_prior_shift")
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}_{}.txt".format(mode, "woshift")
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

        f_preds, m_preds = f_loaded["prediction_sum_prob_w_prior_shift"], m_loaded["prediction_sum_prob_w_prior_shift"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode + "/sum_w_prior_shift")
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}_{}.txt".format(mode, "wshift")
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

        f_preds, m_preds = f_loaded["prediction_max_prob_w_prior_shift"], m_loaded["prediction_max_prob_w_prior_shift"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode + "/max_w_prior_shift")
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}_{}.txt".format(mode, "max")
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

    elif mode == "gradproj_adv":
        f_preds, m_preds = f_loaded["class_predict_labels"], m_loaded["class_predict_labels"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode)
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}.txt".format(mode)
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

    elif mode == "uniconf_adv":
        f_preds, m_preds = f_loaded["class_predict_labels"], m_loaded["class_predict_labels"]
        disps = []
        for c in range(8):
            cnt_f, cnt_m = 0, 0
            for f_pred, m_pred in zip(f_preds, m_preds):
                if f_pred == c:
                    cnt_f +=1
                if m_pred == c:
                    cnt_m +=1
            disp = max(cnt_f, cnt_m) / (cnt_f + cnt_m) 
            disps.append(disp)
        print(mode)
        print(m_load_path)
        print("disps:{}".format(disps))
        av_disp = sum(disps) / len(disps)
        diff = max(disps) - (sum(disps)/len(disps))
        save_path = "record/imdb_eval_result/{}.txt".format(mode)
        with open(save_path, "a") as f:
            f.write(str(av_disp) + ", " + str(diff) + "\n")

if __name__ == "__main__":
    main()