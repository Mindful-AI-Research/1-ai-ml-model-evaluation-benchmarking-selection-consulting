1) Prediction error is defined as:

* A) predicted value − mean of the values
* B) actual value − predicted value ✅
* C) |actual value| ÷ predicted value
* D) always a number between 0 and 1

Answer: B) actual value − predicted value.

#

2) MAE (Mean Absolute Error) is:

* A) the average of the errors without their signs (absolute values) ✅
* B) the square root of the mean squared errors
* C) the error expressed as a percentage
* D) the variance of the actual values

Answer: A) the average of the errors without their signs (absolute values).

#

3) Why is RMSE more sensitive to large errors than MAE?

* A) because it divides each error by the actual value
* B) because it ignores small errors
* C) because it uses the median instead of the mean
* D) because it squares each error before summing ✅

Answer: D) because it squares each error before summing.

#

4) Regarding the range of R², which statement is correct?

* A) it always ranges from 0 to 1
* B) it is measured in the same units as the target
* C) it can be at most 1 and can be negative ✅
* D) the lower, the better the model

Answer: C) it can be at most 1 and can be negative.

#

5) A negative R² means that the model:

* A) predicted all values correctly
* B) performs worse than simply always predicting the mean ✅
* C) has zero error
* D) is the best possible model

Answer: B) performs worse than simply always predicting the mean.

#

6) MAPE expresses the error:

* A) in the same units as the target
* B) squared
* C) as a fraction of the variance
* D) as a percentage of the actual value ✅

Answer: D) as a percentage of the actual value.

⸻

7) The main pitfall of MAPE occurs when:

* A) the actual values are very large
* B) the target is categorical (yes/no)
* C) there are actual values close to zero ✅
* D) the model is linear

Answer: C) there are actual values close to zero.

#

8) To compare two models that predict targets with different units/scales (e.g., minutes vs. currency), it is more appropriate to use:

* A) MAE
* B) RMSE
* C) maximum error in units
* D) R² or MAPE (unitless metrics) ✅

Answer: D) R² or MAPE (unitless metrics).

#

9) In the calculation of R², the “sum of squared deviations from the mean” represents:

* A) how much the actual values spread around the mean (the variance) ✅
* B) the model’s error for each prediction
* C) the predicted mean value
* D) the MAPE of the dataset

Answer: A) how much the actual values spread around the mean.

#

💡 Relevant question for the class

“In a regression problem with actual values close to zero and some very large errors, which combination of metrics would be most appropriate for evaluating the model without allowing a single metric to distort the interpretation of its performance?”
