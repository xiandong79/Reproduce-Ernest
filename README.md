# Reproduce-Ernest


# Implementation

## Step-1 Seclect Most informative points

![](http://wx2.sinaimg.cn/mw690/006BEqu9gy1fgxgq8jyamj30mo0bf425.jpg)

## Step-2 Get Runtimes from ec2
![](http://wx1.sinaimg.cn/mw690/006BEqu9gy1fgxgq8o3d7j306e053aab.jpg)

## Step-3 Train Model and Predict
![](http://wx2.sinaimg.cn/mw690/006BEqu9gy1fgxgq8mjtjj30gh0cbglz.jpg)

# Ernest Design Philosophy

## Background
- more and more `machine learning/analytic` jobs
- hard to choose cloud configration 

## Motivation
- we need `a prediction` without history
- we need `a prediction` with low cost

## Design 

### Model and Why 


> why we can use this model?

The machine learning jobs have **common structure** ~ 
1. iteration. As a result, we can only run `a few` iteration.
2. In each iteration, the `shuffle`, `join`, `collect` is similar in all machine jobs.  


### How to select 

> Why we use `Optimal expermient design` this technique?

If we are the 

The advantages of `optimal expermient` is:

1. low cost
2. 

### Prediction

> Why we use NNLS instead of linear regression?

## Summary

## Advantages

## Drawbacks


## Revised-Kmeans Example

This example is based on Spark-Bench.

The file which need to be revised are included in this folder.