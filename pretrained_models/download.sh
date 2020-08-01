#!/bin/bash
BASE_URL="https://learner.csie.ntu.edu.tw/%7Ecybai/tc_models/"
for f in ConvLSTM.zip ConvLSTM_CCA.zip ConvLSTM_SSA.zip ConvLSTM_CCA_SSA.zip
do
  wget $BASE_URL$f --no-check-certificate
done
