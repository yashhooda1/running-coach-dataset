# sources/

Inputs the build reads. Not generated.

    Getting_back_into_competitive_running.txt
    Advanced_Training_Plan_for_sub_30_8k.txt
    Sub_120_half_marathon_training_plan.txt
    Sub_450_Mile_Training_Plan_Updated.txt
    activities.csv        <- Strava bulk export, gitignored

`parse_plans.py` globs every `*.txt` here, so adding a fifth plan needs no code
change. `activities.csv` is optional: without it, `build.py` skips the
`training_review` split and the calibration constants.

To get activities.csv: Strava > Settings > My Account > Download or Delete Your
Account > Request Your Archive. It arrives as a zip; this is the file at its root.
