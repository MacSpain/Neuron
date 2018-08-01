#define _CRT_SECURE_NO_WARNINGS

#include "Header.h"
#include "FileNames.h"

float TrainNetwork(neural_network *Network, float *NeuronValues, int Type)
{
	for (int InputNeuronIndex = 0;
		InputNeuronIndex < Network->InputNeuronCount;
		++InputNeuronIndex)
	{
		Network->InputNeurons[InputNeuronIndex] = NeuronValues[InputNeuronIndex];
	}

	float **HiddenWeightedSums = (float **)malloc(Network->HiddenLayerCount*sizeof(float *));

	for (int HiddenLayerIndex = 0;
		HiddenLayerIndex < Network->HiddenLayerCount;
		++HiddenLayerIndex)
	{
		neuron_layer *HiddenLayer = Network->HiddenLayers + HiddenLayerIndex;
		neuron_layer *PreviousLayer = 0;

		int WeightCount = 0;

		if (HiddenLayerIndex == 0)
		{
			WeightCount = Network->InputNeuronCount;
		}
		else
		{
			WeightCount = Network->HiddenLayers[HiddenLayerIndex - 1].NeuronCount;
			PreviousLayer = Network->HiddenLayers + HiddenLayerIndex - 1;
		}

		HiddenWeightedSums[HiddenLayerIndex] = (float *)malloc(HiddenLayer->NeuronCount*sizeof(float));
		float *WeightedSums = HiddenWeightedSums[HiddenLayerIndex];

		for (int NeuronIndex = 0;
			NeuronIndex < HiddenLayer->NeuronCount;
			++NeuronIndex)
		{
			float WeightedSum = 0.0f;
			float *NeuronWeights = HiddenLayer->Weights[NeuronIndex];

			for (int WeightIndex = 0;
				WeightIndex < WeightCount;
				++WeightIndex)
			{
				if (PreviousLayer)
				{
					WeightedSum += PreviousLayer->Neurons[WeightIndex] * NeuronWeights[WeightIndex];
				}
				else
				{
					WeightedSum += Network->InputNeurons[WeightIndex] * NeuronWeights[WeightIndex];
				}
			}

			//float BiasedSum = WeightedSum + HiddenLayer->Biases[NeuronIndex];

			WeightedSums[NeuronIndex] = Sigmoid(WeightedSum);
			
			HiddenLayer->Neurons[NeuronIndex] = Sigmoid(WeightedSum);
		}
	}

	float *OutputWeightedSums = (float *)malloc(Network->OutputNeuronCount*sizeof(float));

	neuron_layer *PreviousLayer = 0;
	float *InputNeurons = Network->InputNeurons;
	int WeightCount;
	if (Network->HiddenLayerCount == 0)
	{
		WeightCount = Network->InputNeuronCount;
	}
	else
	{
		PreviousLayer = Network->HiddenLayers + (Network->HiddenLayerCount - 1);
		WeightCount = PreviousLayer->NeuronCount;
	}

	for (int OutputNeuronIndex = 0;
		OutputNeuronIndex < Network->OutputNeuronCount;
		++OutputNeuronIndex)
	{
		float WeightedSum = 0.0f;
		float *NeuronWeights = *(Network->OutputNeurons.Weights + OutputNeuronIndex);

		for (int WeightIndex = 0;
			WeightIndex < WeightCount;
			++WeightIndex)
		{
			if (PreviousLayer)
			{
				WeightedSum += PreviousLayer->Neurons[WeightIndex] * NeuronWeights[WeightIndex];
			}
			else
			{
				WeightedSum += InputNeurons[WeightIndex] * NeuronWeights[WeightIndex];
			}
		}

		//float BiasedSum = WeightedSum + Network->OutputNeurons.Biases[OutputNeuronIndex];

		OutputWeightedSums[OutputNeuronIndex] = Sigmoid(WeightedSum);

		Network->OutputNeurons.Neurons[OutputNeuronIndex] = Sigmoid(WeightedSum);
	}

	float CostVector[2] = { 0.0f, 0.0f };
	float ExpectedCat = 0.0f;
	float ExpectedDog = 0.0f;

	if (Type == CAT)
	{
		ExpectedCat = 1.0f;
	}
	else
	{
		ExpectedDog = 1.0f;
	}

	float Error = 0.0f;
	Error += (1.0f / 2.0f)*((ExpectedCat - Network->OutputNeurons.Neurons[0])*(ExpectedCat - Network->OutputNeurons.Neurons[0]));
	Error += (1.0f / 2.0f)*((ExpectedDog - Network->OutputNeurons.Neurons[1])*(ExpectedDog - Network->OutputNeurons.Neurons[1]));

	CostVector[0] = (Network->OutputNeurons.Neurons[0] - ExpectedCat);
	CostVector[1] = (Network->OutputNeurons.Neurons[1] - ExpectedDog);

	float **Gradients = 0;
	int NeuronCount;
	if (PreviousLayer)
	{
		Gradients = (float **)malloc(Network->HiddenLayerCount*sizeof(float *));
		for (int HiddenLayerIndex = 0;
			HiddenLayerIndex < Network->HiddenLayerCount;
			++HiddenLayerIndex)
		{
			if (HiddenLayerIndex == 0)
			{
				Gradients[HiddenLayerIndex] = (float *)malloc(PreviousLayer->NeuronCount*sizeof(float));
			}
			else
			{
				Gradients[HiddenLayerIndex] = (float *)malloc(Network->HiddenLayers[HiddenLayerIndex - 1].NeuronCount*sizeof(float));
			}
		}
		NeuronCount = PreviousLayer->NeuronCount;
	}
	else
	{
		Gradients = (float **)malloc(sizeof(float *));
		Gradients[0] = (float *)malloc(Network->InputNeuronCount*sizeof(float));
		NeuronCount = Network->InputNeuronCount;
	}

	for (int PreviousNeuronIndex = 0;
		PreviousNeuronIndex < NeuronCount;
		++PreviousNeuronIndex)
	{
		if (Network->HiddenLayerCount == 0)
		{
			Gradients[0][PreviousNeuronIndex] = 0.0f;

			for (int OutputNeuronIndex = 0;
				OutputNeuronIndex < Network->OutputNeuronCount;
				++OutputNeuronIndex)
			{
				Gradients[0][PreviousNeuronIndex] += Network->OutputNeurons.Weights[OutputNeuronIndex][PreviousNeuronIndex] * SigmoidDerivative(OutputWeightedSums[OutputNeuronIndex]) *CostVector[OutputNeuronIndex];
			}
		}
		else
		{
			Gradients[Network->HiddenLayerCount - 1][PreviousNeuronIndex] = 0.0f;

			for (int OutputNeuronIndex = 0;
				OutputNeuronIndex < Network->OutputNeuronCount;
				++OutputNeuronIndex)
			{
				Gradients[Network->HiddenLayerCount - 1][PreviousNeuronIndex] += Network->OutputNeurons.Weights[OutputNeuronIndex][PreviousNeuronIndex] * SigmoidDerivative(OutputWeightedSums[OutputNeuronIndex]) *CostVector[OutputNeuronIndex];
			}

		}
	}

	for (int HiddenLayerIndex = Network->HiddenLayerCount - 2;
		HiddenLayerIndex >= 0;
		--HiddenLayerIndex)
	{
		neuron_layer *CurrentLayer = Network->HiddenLayers + HiddenLayerIndex + 1;

		int PreviousLayerNeuronCount = Network->HiddenLayers[HiddenLayerIndex].NeuronCount;
		int CurrentLayerNeuronCount = CurrentLayer->NeuronCount;

		for (int PreviousNeuronIndex = 0;
			PreviousNeuronIndex < PreviousLayerNeuronCount;
			++PreviousNeuronIndex)
		{
			Gradients[HiddenLayerIndex][PreviousNeuronIndex] = 0.0f;

			for (int CurrentNeuronIndex = 0;
				CurrentNeuronIndex < CurrentLayerNeuronCount;
				++CurrentNeuronIndex)
			{
				Gradients[HiddenLayerIndex][PreviousNeuronIndex] += CurrentLayer->Weights[CurrentNeuronIndex][PreviousNeuronIndex] * SigmoidDerivative(HiddenWeightedSums[HiddenLayerIndex + 1][CurrentNeuronIndex])*Gradients[HiddenLayerIndex + 1][CurrentNeuronIndex];
			}
		}
	}

	for (int OutputNeuronIndex = 0;
		OutputNeuronIndex < Network->OutputNeuronCount;
		++OutputNeuronIndex)
	{

		float dSigmoid = SigmoidDerivative(OutputWeightedSums[OutputNeuronIndex]);

		float *NeuronWeights = *(Network->OutputNeurons.Weights + OutputNeuronIndex);

		for (int WeightIndex = 0;
			WeightIndex < NeuronCount;
			++WeightIndex)
		{
			if (PreviousLayer)
			{
				NeuronWeights[WeightIndex] -= PreviousLayer->Neurons[WeightIndex] * dSigmoid * CostVector[OutputNeuronIndex];
			}
			else
			{
				NeuronWeights[WeightIndex] -= InputNeurons[WeightIndex] * dSigmoid * CostVector[OutputNeuronIndex];
			}
		}

		//Network->OutputNeurons.Biases[OutputNeuronIndex] += CostVector[OutputNeuronIndex];
	}

#if 0
	for (int PreviousNeuronIndex = 0;
		PreviousNeuronIndex < PreviousLayer->NeuronCount;
		++PreviousNeuronIndex)
	{
		float SigmoidTimesCost = 0.0f;
		for (int OutputNeuronIndex = 0;
			OutputNeuronIndex < Network->OutputNeuronCount;
			++OutputNeuronIndex)
		{
			SigmoidTimesCost += Sigmoids[OutputNeuronIndex] * CostVector[OutputNeuronIndex];
		}

		SumsOfWeightsFromOutputLayer[PreviousNeuronIndex] *= SigmoidTimesCost;
	}
#endif
	
	for (int HiddenLayerIndex = Network->HiddenLayerCount - 1;
		HiddenLayerIndex >= 0;
		--HiddenLayerIndex)
	{
		if (HiddenLayerIndex == 0)
		{
			float *PreviousLayer = Network->InputNeurons;

			for (int HiddenNeuronIndex = 0;
				HiddenNeuronIndex < Network->HiddenLayers[HiddenLayerIndex].NeuronCount;
				++HiddenNeuronIndex)
			{

				float *WeightedSums = HiddenWeightedSums[HiddenLayerIndex];
				float dSigmoid = SigmoidDerivative(WeightedSums[HiddenNeuronIndex]);

				float *NeuronWeights = Network->HiddenLayers[HiddenLayerIndex].Weights[HiddenNeuronIndex];

				for (int WeightIndex = 0;
					WeightIndex < Network->InputNeuronCount;
					++WeightIndex)
				{
					NeuronWeights[WeightIndex] -= PreviousLayer[WeightIndex]*dSigmoid*Gradients[HiddenLayerIndex][HiddenNeuronIndex];
				}

				//Network->HiddenLayers[HiddenLayerIndex].Biases[HiddenNeuronIndex] += GradientsFromOutputLayer[HiddenNeuronIndex];
			}
		}
	}

	for (int HiddenWeightedSumIndex = 0;
		HiddenWeightedSumIndex < Network->HiddenLayerCount;
		++HiddenWeightedSumIndex)
	{
		free(HiddenWeightedSums[HiddenWeightedSumIndex]);
	}
	free(HiddenWeightedSums);
	free(OutputWeightedSums);
	for (int HiddenWeightedSumIndex = 0;
		HiddenWeightedSumIndex < Network->HiddenLayerCount;
		++HiddenWeightedSumIndex)
	{
		free(Gradients[HiddenWeightedSumIndex]);
	}
	free(Gradients);

	return Error;
}

bool TestNetwork(neural_network *Network, float *NeuronValues, int Type)
{
	for (int InputNeuronIndex = 0;
		InputNeuronIndex < Network->InputNeuronCount;
		++InputNeuronIndex)
	{
		Network->InputNeurons[InputNeuronIndex] = NeuronValues[InputNeuronIndex];
	}

	for (int HiddenLayerIndex = 0;
		HiddenLayerIndex < Network->HiddenLayerCount;
		++HiddenLayerIndex)
	{
		neuron_layer *HiddenLayer = Network->HiddenLayers + HiddenLayerIndex;
		neuron_layer *PreviousLayer = 0;

		int WeightCount = 0;

		if (HiddenLayerIndex == 0)
		{
			WeightCount = Network->InputNeuronCount;
		}
		else
		{
			WeightCount = Network->HiddenLayers[HiddenLayerIndex - 1].NeuronCount;
			PreviousLayer = Network->HiddenLayers + HiddenLayerIndex - 1;
		}

		for (int NeuronIndex = 0;
			NeuronIndex < HiddenLayer->NeuronCount;
			++NeuronIndex)
		{
			float WeightedSum = 0.0f;
			float *NeuronWeights = *(HiddenLayer->Weights + NeuronIndex);

			for (int WeightIndex = 0;
				WeightIndex < WeightCount;
				++WeightIndex)
			{
				if (PreviousLayer)
				{
					WeightedSum += PreviousLayer->Neurons[WeightIndex] * NeuronWeights[WeightIndex];
				}
				else
				{
					WeightedSum += Network->InputNeurons[WeightIndex] * NeuronWeights[WeightIndex];
				}
			}

			//float BiasedSum = WeightedSum + HiddenLayer->Biases[NeuronIndex];

			HiddenLayer->Neurons[NeuronIndex] = Sigmoid(WeightedSum);
		}
	}

	float *PreviousNeurons;
	int WeightCount;
	if (Network->HiddenLayerCount == 0)
	{
		PreviousNeurons = Network->InputNeurons;
		WeightCount = Network->InputNeuronCount;
	}
	else
	{
		PreviousNeurons = Network->HiddenLayers[Network->HiddenLayerCount - 1].Neurons;
		WeightCount = Network->HiddenLayers[Network->HiddenLayerCount - 1].NeuronCount;
	}

	for (int OutputNeuronIndex = 0;
		OutputNeuronIndex < Network->OutputNeuronCount;
		++OutputNeuronIndex)
	{
		float WeightedSum = 0.0f;
		float *NeuronWeights = *(Network->OutputNeurons.Weights + OutputNeuronIndex);

		for (int WeightIndex = 0;
			WeightIndex < WeightCount;
			++WeightIndex)
		{
			WeightedSum += PreviousNeurons[WeightIndex] * NeuronWeights[WeightIndex];
		}

		//float BiasedSum = WeightedSum + Network->OutputNeurons.Biases[OutputNeuronIndex];

		Network->OutputNeurons.Neurons[OutputNeuronIndex] = Sigmoid(WeightedSum);
	}

	if (Type == CAT)
	{
		if (Network->OutputNeurons.Neurons[0] > Network->OutputNeurons.Neurons[1])
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		if (Network->OutputNeurons.Neurons[1] > Network->OutputNeurons.Neurons[0])
		{
			return true;
		}
		else
		{
			return false;
		}
	}
}

int main()
{
	FILE *csv = fopen("wyniki.txt", "w");

	fprintf(csv, "Trening;Sredni blad;Wynik walidacji\n");

	srand(time(0));

	int HiddenLayerNeuronCounts[] = { FREQUENCY_BAND_COUNT / 2, FREQUENCY_BAND_COUNT / 2 };

	neural_network Network = SetupNeuralNetwork(FREQUENCY_BAND_COUNT, 2, 2, HiddenLayerNeuronCounts);

	//Tablica wartosci wprowadzanych do neuronow sieci
	float *NeuronValuesTab = (float *)malloc(sizeof(float) * FREQUENCY_BAND_COUNT);

	//Inicjalizacja wartosci do transformaty Fouriera
	kiss_fftr_cfg cfg = kiss_fftr_alloc(FFT_WINDOW_SIZE, 0, 0, 0);
	int OutCount = (FFT_WINDOW_SIZE / 2 + 1);
	float *SamplesTab = (float *)malloc(sizeof(float) * FFT_WINDOW_SIZE);
	kiss_fft_cpx *out = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx)*OutCount);

	//Ilosc plikow do trenowania sieci
	int TrainCountDogs = ArrayCount(DogsTrainFiles)*0.6f;
	int TrainCountCats = ArrayCount(CatsTrainFiles)*0.6f;

	int ValidationCountDogs = (ArrayCount(DogsTrainFiles) - TrainCountDogs)*0.5f;
	int ValidationCountCats = (ArrayCount(CatsTrainFiles) - TrainCountCats)*0.5f;

	//Wynik walidacji i granica walidacji
	float ValidationResult = 0.0f; 
	float ValidationThreshold = 0.9f;

	//Tablice dostepu do plikow treningowych, true oznacza ze plik z 
	//danego indeksu z tablicy plikow treningowych byl juz przetwarzany
	bool *TrainCat = (bool *)malloc(sizeof(bool)*ArrayCount(CatsTrainFiles));
	bool *TrainDog = (bool *)malloc(sizeof(bool)*ArrayCount(DogsTrainFiles));

	int TrainingIndex = 1;

	while (ValidationResult < ValidationThreshold)
	{

		//Zerowanie tablic dostepu do plikow treningowych
		for (int i = 0; i < ArrayCount(CatsTrainFiles); ++i)
		{
			TrainCat[i] = false;
		}

		for (int i = 0; i < ArrayCount(DogsTrainFiles); ++i)
		{
			TrainDog[i] = false;
		}

		int TrainedCats = 0;
		int TrainedDogs = 0;

		float Error = 0.0f;
		//Trenowanie na kotach
		for (int i = 0; i < TrainCountCats + TrainCountDogs; ++i)
		{
			int FileIndex;

			if ((rand() % 2) == 0)
			{
				if (TrainedCats < TrainCountCats)
				{
					++TrainedCats;
					do
					{
						FileIndex = rand() % ArrayCount(CatsTrainFiles);
					} while (TrainCat[FileIndex] != false);

					TrainCat[FileIndex] = true;
					FourierTransform(CatsTrainFiles[FileIndex], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

					Error += TrainNetwork(&Network, NeuronValuesTab, CAT);
				}
				else
				{
					++TrainedDogs;
					do
					{
						FileIndex = rand() % ArrayCount(DogsTrainFiles);
					} while (TrainDog[FileIndex] != false);

					TrainDog[FileIndex] = true;
					FourierTransform(DogsTrainFiles[FileIndex], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

					Error += TrainNetwork(&Network, NeuronValuesTab, DOG);
				}

			}
			else
			{
				if (TrainedDogs < TrainCountDogs)
				{
					++TrainedDogs;
					do
					{
						FileIndex = rand() % ArrayCount(DogsTrainFiles);
					} while (TrainDog[FileIndex] != false);

					TrainDog[FileIndex] = true;
					FourierTransform(DogsTrainFiles[FileIndex], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

					Error += TrainNetwork(&Network, NeuronValuesTab, DOG);
				}
				else
				{
					++TrainedCats;
					do
					{
						FileIndex = rand() % ArrayCount(CatsTrainFiles);
					} while (TrainCat[FileIndex] != false);

					TrainCat[FileIndex] = true;
					FourierTransform(CatsTrainFiles[FileIndex], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

					Error += TrainNetwork(&Network, NeuronValuesTab, CAT);
				}
			}
		}

		Error = Error / (float)(TrainCountCats + TrainCountDogs);
		printf("Sredni blad: %f\n", Error);

		int ValidationCount = 0;
		int ValidationsPassed = 0;

		//Walidacja na kotach
		for (int i = 0; i < ValidationCountCats; ++i)
		{
			int FileIndex;

			do
			{
				FileIndex = rand() % ArrayCount(CatsTrainFiles);
			} while (TrainCat[FileIndex] != false);

			TrainCat[i] = true;
			FourierTransform(CatsTrainFiles[i], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

			++ValidationCount;

			if (TestNetwork(&Network, NeuronValuesTab, CAT))
			{
				++ValidationsPassed;
			}

		}

		//Walidacja na psach
		for (int i = 0; i < ValidationCountDogs; ++i)
		{
			int FileIndex;

			do
			{
				FileIndex = rand() % ArrayCount(DogsTrainFiles);
			} while (TrainDog[FileIndex] != false);

			TrainDog[i] = true;
			FourierTransform(DogsTrainFiles[i], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

			++ValidationCount;

			if (TestNetwork(&Network, NeuronValuesTab, DOG))
			{
				++ValidationsPassed;
			}

		}

		ValidationResult = (float)ValidationsPassed / (float)ValidationCount;

		printf("Wynik walidacji: %f\n", ValidationResult*100.0f);
		fprintf(csv, "%d;%f;%f\n", TrainingIndex, Error*100.0f, ValidationResult*100.0f);
		TrainingIndex++;
	}

	int TestCount = 0;
	int TestsPassed = 0;

	//Testowanie na kotach
	for (int i = 0; i < ArrayCount(CatsTrainFiles); ++i)
	{
		if (TrainCat[i] == false)
		{
			FourierTransform(CatsTrainFiles[i], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

			++TestCount;

			if (TestNetwork(&Network, NeuronValuesTab, CAT))
			{
				++TestsPassed;
			}
		}
	}

	//Testowanie na psach
	for (int i = 0; i < ArrayCount(DogsTrainFiles); ++i)
	{
		if (TrainDog[i] == false)
		{
			FourierTransform(DogsTrainFiles[i], cfg, out, NeuronValuesTab, OutCount, SamplesTab, FFT_WINDOW_SIZE, FREQUENCY_BAND_COUNT);

			++TestCount;

			if (TestNetwork(&Network, NeuronValuesTab, DOG))
			{
				++TestsPassed;
			}
		}
	}
	
	float PercentagePassed = (float)TestsPassed / (float)TestCount;

	printf("Procent zdanych testow: %f", PercentagePassed*100.0f);

	fprintf(csv, "Zdane testy;%f", PercentagePassed*100.0f);

	fclose(csv);



	return 0;
}