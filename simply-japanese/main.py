import model.baseline_model





#### Baseline model
"""
if file.exists(baseline model eval):
    skip
else:
    create excel file with original 3 columns
        + predictions
        + wer score
        + bleu score
    for baseline model
"""
if True:
    if os.environ.get("DATA_SOURCE") == "TEST":
        file = os.environ.get("TEST_DATA")
    elif os.environ.get("DATA_SOURCE") == "DEPLOY":
        file = os.environ.get("DEPLOY_DATA")
    else:
        raise Exception ("Data source not or incorrectly specified in env.")
