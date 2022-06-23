# get distribution of data by annotator

import pandas as pd

# get distribution by annotator
def get_dist_by_annotator(annotators_dict):
    """
    Get the distribution of data by annotator
    param annotators_dict : a dict of annotator: [filenames]
    return: distribution of emotion and/or sentiment labels by annotator 
    """
    for annr in annotators_dict.keys():
        files_list = annotators_dict[annr]

        # get dist
        sent, emos = get_dist(files_list)
        
        # print out distribution
        print(f"{annr} sent distribution is:")
        print(sent.value_counts())
        print(f"{annr} emo distribution is:")
        print(emos.value_counts())


def get_dist(files_list):
    """
    Get the distribution for a single set of files
    """
    all_files = None
    for f in files_list:
        pd_df = pd.read_csv(f)
        if all_files is None:
            all_files = pd_df
        else:
            all_files = pd.concat([all_files, pd_df], axis=0)
    
    return all_files['sentiment'], all_files['emotion']


if __name__ == "__main__":

    # import all of CKJ files 
    files_list1 = ['CKJ_HSRData_TrialMessages_Trial-T000603_Team-TM000202_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv', 'CKJ_HSRData_TrialMessages_Trial-T000604_Team-TM000202_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv', 'CKJ_HSRData_TrialMessages_Trial-T000605_Team-TM000203_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv', 'CKJ_HSRData_TrialMessages_Trial-T000607_Team-TM000204_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv']

    # import all of SH files
    files_list2 = ['SH_HSRData_TrialMessages_Trial-T000603_Team-TM000202_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv', 'SH_HSRData_TrialMessages_Trial-T000604_Team-TM000202_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv', 'SH_HSRData_TrialMessages_Trial-T000605_Team-TM000203_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv', 'SH_HSRData_TrialMessages_Trial-T000607_Team-TM000204_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv']

    annr_dict = {"CKJ": files_list1, "SH": files_list2}
