#include "pch.h"
#include <cmath>
#include "Process.h"
float* Add_Data(float* sample, int Size, float* x, int Dim) {
	// Allocate enough space for 'Size' samples, each of dimension 'Dim'
	float* temp = new float[Size * Dim];

	// Copy the old samples into the new array
	for (int i = 0; i < (Size - 1) * Dim; i++) {
		temp[i] = sample[i];
	}

	// Add the new sample at the end (the (Size-1)th sample, zero-indexed)
	for (int i = 0; i < Dim; i++) {
		temp[(Size - 1) * Dim + i] = x[i];
	}

	// Deallocate the old array
	delete[] sample;

	return temp;
}

float* Add_Labels(float* Labels, int Size, int label) {
	// Allocate enough space for 'Size' labels
	float* temp = new float[Size];

	// Copy the old labels into the new array
	for (int i = 0; i < Size - 1; i++) {
		temp[i] = Labels[i];
	}

	// Add the new label at the end
	temp[Size - 1] = static_cast<float>(label);

	// Deallocate the old array
	delete[] Labels;

	return temp;
}

float* init_array_random(int len) {
	float* arr = new float[len];
	for (int i = 0; i < len; i++)
		arr[i] = ((float)rand() / RAND_MAX) - 0.5f;
	return arr;
}
float* init_array_zero(int len) {
	float* arr = new float[len];
	for (int i = 0; i < len; i++)
		arr[i] = 0;
	return arr;
}
void push_back(float*** arr, int lIndex, int wIndex, int size, float item) {
	for (int t = 0; t < size - 1; t++) {
		arr[lIndex][wIndex][t] = arr[lIndex][wIndex][t + 1];
	}
	arr[lIndex][wIndex][size - 1] = item; //w_delta (t-1)
}
float* Batch_Norm(float* Samples, int numSample, int inputDim, float mean[], float variance[], bool copy)
{
	float* normalizedSamples = new float[numSample * inputDim];
	if (copy == true) {


		for (int i = 0; i < inputDim; i++) {
			mean[i] = 0.0f;
			variance[i] = 0.0f;
		}


		for (int i = 0; i < numSample; i++) {
			for (int j = 0; j < inputDim; j++) {
				// Add each sample's j-th dimension value to mean[j],
				// casting to float if needed:
				mean[j] += static_cast<float>(Samples[i * inputDim + j]);
			}
		}

		// After this loop completes, you’ll divide each mean[j] by numSample:
		for (int j = 0; j < inputDim; j++) {
			mean[j] /= numSample;
		}



		for (int i = 0; i < numSample; i++) {
			for (int j = 0; j < inputDim; j++) {
				float diff = Samples[i * inputDim + j] - mean[j];
				variance[j] += diff * diff;  // purely float operations, no pow()
			}
		}

		for (int j = 0; j < inputDim; j++) {
			variance[j] /= numSample;
		}



		// Assuming normalizedSamples is already allocated with size numSample * inputDim
		for (int i = 0; i < numSample; i++) {
			for (int j = 0; j < inputDim; j++) {
				normalizedSamples[i * inputDim + j] = (Samples[i * inputDim + j] - mean[j]) / sqrt(variance[j]);
			}
		}

	}
	else
	{
		for (int i = 0; i < numSample; i++)
			for (int j = 0; j < inputDim; j++)
				normalizedSamples[i * inputDim + j] = (Samples[i * inputDim + j] - mean[j]) / sqrt(variance[j]);
	}

	return normalizedSamples;
}
int YPoint(int x, double* w, int carpan) {
	if (w[1] == 0) return INT_MAX; // Bölme hatasý önlenir
	return (int)((double)(-1 * (double)carpan * w[2] - w[0] * x) / (double)(w[1]));
};