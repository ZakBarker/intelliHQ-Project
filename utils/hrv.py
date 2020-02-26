import numpy as np

def Calculate_Features(R_peaks, fs=1, decim=2):
    R_peaks = np.array(R_peaks)
    # Calculating SDNN
    R_peaks2 = R_peaks / fs  # Turn R-Peak locations to time stamps
    R_td = np.diff(R_peaks2)
    MeanRR = np.round(np.mean(R_td) * 1e3, decim)
    SDNN = np.round(np.std(R_td) * 1e3, decim)

    # Calculating SDANN
    timejump = 300  # 5 minutes
    timestamp = timejump
    runs = int(R_peaks2[-1] / timestamp)
    SDNN_5 = np.zeros(runs)
    i = 0
    while (timestamp <= timejump * runs):
        section = R_peaks2[R_peaks2 <= timestamp]
        R_peaks2 = R_peaks2[R_peaks2 > timestamp]
        timestamp += timejump
        R_td_5 = np.diff(section)
        SDNN_5[i] = np.std(R_td_5)
        i += 1

    SDANN = np.round(np.mean(SDNN_5) * 1e3, decim)

    # Calculating pNN50                      pNN50 = (NN50 count) / (total NN count)
    total_NN = len(R_peaks)
    NN_50 = np.diff(R_td)
    count = 0
    for i in range(len(NN_50)):
        if NN_50[i] > 0.050:
            count = count + 1

    pNN50 = np.round((count / total_NN * 100), decim)

    # Calculating RMSSD

    R_peaks3 = R_peaks / fs * 1000  # Turn R-Peak locations to time stamps
    RMSSD = np.round((np.sqrt(np.sum(np.power(np.diff(R_peaks3), 2)) / (len(R_peaks3) - 1))), decim)

    return (SDNN, SDANN, MeanRR, RMSSD, pNN50)