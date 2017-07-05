#!/usr/bin/env python
# coding=utf-8
import pandas as pd

result_pd = pd.read_csv('./Submission.csv')

predictions = result_pd['predictions']

st = -1
for i in xrange(len(predictions)):
    if predictions[i] == 1:
        st = i
        break

ed = -1
for i in xrange(len(predictions)-1,-1,-1):
    if predictions[i] == 1:
        ed = i
        break
t1, t2 = [st+1], [ed+1]
sub = pd.DataFrame({'t1':t1,'t2':t2})
sub.to_csv('final_sub.csv', index=False)
print 'result has been saved successfully!'

