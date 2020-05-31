import utils

mode = "domain_discriminative"
m_load_path = "record/imdb_domain_discriminative/cifar_color/test_male_result.pkl"
f_load_path = "record/imdb_domain_discriminative/cifar_color/test_female_result.pkl"

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

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
    print("disps:{}".format(disps))
    print("av_disp:{}".format(sum(disps)/len(disps)))

