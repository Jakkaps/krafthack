from typing import overload
import numpy as np
from math import floor
from sklearn.linear_model import LinearRegression

class LinearScaler(LinearRegression):

	def scale_by_bucket_mean(self, x, y, n_buckets):
		scale = self.predict(x)
		y = np.copy(y)
		offset = floor(len(y) / n_buckets)
		
		for i in range(0, len(y), offset):
			scale_bucket_mean = np.mean(scale[i:i+offset])
			original_bucket_mean = np.mean(y[i:i+offset])

			if (scale_bucket_mean - original_bucket_mean > 10):
				continue


			y[i:i+offset] = y[i:i+offset] * scale_bucket_mean / original_bucket_mean
		
		return y