# Reproduce-Ernest


# Implementation

## Step-1 Seclect Most informative points

![](http://wx2.sinaimg.cn/mw690/006BEqu9gy1fgxgq8jyamj30mo0bf425.jpg)

The above figure shows that result a example tested on EC2-m4.large which is equiped by 2 cores per machine. 

## Step-2 Get Runtimes from ec2
![](http://wx1.sinaimg.cn/mw690/006BEqu9gy1fgxgq8o3d7j306e053aab.jpg)

This is the runtime of selected training points according to the example given by Ernest paper.

## Step-3 Train Model and Predict
Using the runtime of selected training points above, we train the final model by `scipy.optimize.nnls` and predict the runtime of test-points.

![](http://wx2.sinaimg.cn/mw690/006BEqu9gy1fgxgq8mjtjj30gh0cbglz.jpg)



# Ernest Design Philosophy

## Background
- hard to choose cloud configration 
- more and more `machine learning/analytic` jobs

## Why previous method is not suitable ?

- Based on historical data, not always hold.
	- data size changes
	- data content changes 
- detailed parametric model
	- good for two-stage MR jobs, but not for complex jobs in Spark  
- we need `a prediction` without history
- we need `a prediction` with low cost

## Design details 


### 1. Model and Why 


> why use this model?

We only model `machine learning/analytic` jobs.
![](http://wx1.sinaimg.cn/mw690/006BEqu9gy1fgxo7bzvscj30m008k757.jpg)

> why this features?

The machine learning jobs have **common structure**. As a result, this model contains 4 features. 
1. serial compuation, a constant $(1)$ is hiddened.
2. parallel computation, $(sclae/machines)$.
3. a $log(machines)$ models communication like aggregation.
4. a $(machines)$ models all-to-one communication.

### 2. How to select

If we are the practitioners and have **no or 0** information about the model we want, definitely, we want the most informative data points so that we can spend training cost as little as possible.

> Why `Optimal expermient design` a statistical technique?

The advantages of optimal expermient expermient is:

1. low cost with selecting the most useful points first.
2. accommodate multiple types of factors, such as process, mixture, and discrete factors.
3. Designs can be optimized when the design-space is constrained

> why this model can use Optimal expermient design?

1. the training points waiting to be selected is in a constrained space. (i.e. data sets range from 1% to 10%, machines range from 1 to 8, then we have 80 feature vectors)

### 3. Prediction

> Why NNLS instead of linear regression?

1. NNLS ensures that each term contributes some **non-negative** amount to the overall time taken. 
2. NNLS avoids over-fitting and also avoids corner cases where say the running time could become negative.

# Summary

## Advantages
- reduce the `modeling` cost by `optimal design` for one application.
- need not to know the inner structure of each job.
- Without using historical data, suitable for various machine learning jobs and complex hardward environment.


## Drawbacks
- Only suitable for machine learning jobs, the limition of `linear model`.
	- not suitable for SQL jobs with selectivity clauses
- One test only suitable for one application and one type of machine, huge cost actually.
- hard to sample a representative dataset for small-scale experiment.
- Waste the `useful` historical data. More than 60% are recurring jobs.
- Good for long running jobs, training cost is too high for short term jobs. 
- Including **hyperparameter** (i.e. $\lambda$ - selection threshold), which has to be finetuned for each application.
- does not consider the `whole process`, i.e. JVM warm-up.


## Revised-Kmeans Example
The scripts need to be revised are included in this folder.