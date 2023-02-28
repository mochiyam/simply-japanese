import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def bleu_score_sentence(reference, target):
    """
    Takes two strings of data, evaluates the bleu score and returns it as a float
    Input:
    reference (string) = "correct" translation as reference
    target (string) = translated text (MT)
    Output:
    bleu (float) = evaluation
    """
    return sentence_bleu([reference], target)

def bleu_score_series(source, reference, target):
    """
    Takes two series of data, evaluates the bleu score and returns it as a float
    Input:
    reference (series) = "correct" translation as reference
    target (series) = translated text (MT)
    Output:
    bleu (float) = evaluation
    """
    len(reference) == len(target)
    bleu_list = []

    for i in range(len(source)):
        bleu_list.append(bleu_score_sentence(reference[i], target[i]))


    bleu = pd.Series(bleu_list)
    return bleu

def wer_score(predicted, simplified, debug=True):
    '''
    Compares the simplified ML prediction of a given text to the pre-existing simplified
    text given with the dataframe.
    Using the WER (word error rate) algorithm.
    Adds the WER score as a new column to the Dataframe
    '''
    r = predicted.split()
    h = simplified.split()
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)


def wer_jp(original, simplified):
    ori = ''
    simpi = ''
    for i in original:
        ori += i + ' '
    for i in simplified:
        simpi += i + ' '
    return round(wer_score(ori, simpi),3)


def evaluate_wer_score(df, c1, c2, score):
    wer_list = []
    for i in df.index:
        original_text = df.iloc[i][c1]
        simplified_text = df.iloc[i][c2]
        wer_list.append(wer_jp(original_text, simplified_text))
    df[score] = wer_list

    return df

def evaluate_blue_score(df, c1, c2, score):
    blue_list = []
    for i in df.index:
        original_text = df.iloc[i][c1]
        simplified_text = df.iloc[i][c2]
        blue_list.append(bleu_score_sentence(original_text, simplified_text))
    df[score] = blue_list

    return df
