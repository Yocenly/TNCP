# TNCP
This repository is created for our paper.
## 1. Experimental Results Download
Due to the size limitation, our experimental results can be achieved on Google Drive [recorder](http://www.baidu.com).
After the download of the package, we ask the directory list should be formed as bellow:

    TNCP
    ----datasets
    ----figures
    ----recorders
    ----the other files

If you do not use our experimental results, please build a new folder named **recorders** which contains following empty folders:

    recorders
    ----ATNC
    ----KNM
    ----RED
    ----RND
    ----SV
    ----TNC

Then the **recorders** folder should be added into the root directory shown as the above.

## 2. Introductions of Code Files
> **base_*.py**: Some basic definitions and basic methods/variables are list there;<br/>
> **method_*.py**: Source files of our proposed method, e.i., TNC and ATNC, as well as the baseline methods, e.i., RND, RED, KNM and SV.<br/>
> **calculate_*.py**: We use them to calculate the metrics mentioned in our paper.<br/>
> **visualize_*.py**: We use them to visualize the figures shown in our paper.

## 3. Run Our Code
You can run our .py files directly in the console. For example, if you want to test the ATNC method, then
    
    python method_atnc.py

Then, in order to see the metrics achieved by the results, we can 

    python calculate_metrics.py
