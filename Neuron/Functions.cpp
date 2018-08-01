#define _CRT_SECURE_NO_WARNINGS
#include "Header.h"

entire_file
ReadEntireFile(char *FileName)
{
	entire_file Result = {};

	FILE *In = fopen(FileName, "rb");
	if (In)
	{
		fseek(In, 0, SEEK_END);
		Result.ContentsSize = ftell(In);
		fseek(In, 0, SEEK_SET);

		Result.Contents = malloc(Result.ContentsSize);
		fread(Result.Contents, Result.ContentsSize, 1, In);
		fclose(In);
	}
	else
	{
		printf("ERROR: Cannot open file %s. \n", FileName);
	}
	return Result;
}

inline riff_iterator
ParseChunkAt(void *At, void *Stop)
{
	riff_iterator Iter;

	Iter.At = (unsigned char *)At;
	Iter.Stop = (unsigned char *)Stop;

	return Iter;
}

inline riff_iterator
NextChunk(riff_iterator Iter)
{
	WAVE_chunk *Chunk = (WAVE_chunk *)Iter.At;
	unsigned int Size = (Chunk->Size + 1) & ~1;
	Iter.At += sizeof(WAVE_chunk) + Size;

	return Iter;
}

inline bool
IsValid(riff_iterator Iter)
{
	bool Result = (Iter.At < Iter.Stop);

	return Result;
}

inline void *
GetChunkData(riff_iterator Iter)
{
	void *Result = (Iter.At + sizeof(WAVE_chunk));

	return Result;
}

inline unsigned int
GetChunkDataSize(riff_iterator Iter)
{
	WAVE_chunk *Chunk = (WAVE_chunk *)Iter.At;
	unsigned int Result = Chunk->Size;

	return Result;
}

inline unsigned int
GetType(riff_iterator Iter)
{
	WAVE_chunk *Chunk = (WAVE_chunk *)Iter.At;
	unsigned int Result = Chunk->ID;

	return Result;
}

loaded_sound
LoadWAV(char *FileName)
{
	loaded_sound Result = {};

	entire_file ReadResult = ReadEntireFile(FileName);
	if (ReadResult.ContentsSize != 0)
	{
		Result.Free = ReadResult.Contents;

		WAVE_header *Header = (WAVE_header *)ReadResult.Contents;

		unsigned int ChannelCount = 0;
		unsigned int SampleDataSize = 0;
		short *SampleData = 0;
		for (riff_iterator Iter = ParseChunkAt(Header + 1, (unsigned char *)(Header + 1) + Header->Size - 4);
			IsValid(Iter);
			Iter = NextChunk(Iter))
		{
			switch (GetType(Iter))
			{
			case WAVE_ChunkID_fmt:
			{
				WAVE_fmt *fmt = (WAVE_fmt *)GetChunkData(Iter);
				Result.SamplesPerSec = fmt->nSamplesPerSec;

				ChannelCount = fmt->nChannels;
			} break;

			case WAVE_ChunkID_data:
			{
				SampleData = (short *)GetChunkData(Iter);
				SampleDataSize = GetChunkDataSize(Iter);
			} break;
			}
		}


		Result.ChannelCount = ChannelCount;
		unsigned int SampleCount = SampleDataSize / (ChannelCount*sizeof(short));
		Result.Time = SampleCount / Result.SamplesPerSec;
		if (ChannelCount == 1)
		{
			Result.Samples = SampleData;
		}

		bool AtEnd = true;
		Result.ChannelCount = 1;



		if (!Result.SampleCount)
		{
			Result.SampleCount = SampleCount;
		}
	}
	return Result;
}


void FourierTransform(char *FileName, kiss_fftr_cfg &cfg, kiss_fft_cpx *out, float *tab, int OutCount, float *SamplesTab, int FFTWindowSize, int FrequencyBandCount)
{
	loaded_sound Sound = LoadWAV(FileName);

	int SampleCount = (Sound.SampleCount % 2) ? (Sound.SampleCount - 1) : Sound.SampleCount;

	int IterationCount = SampleCount / FFTWindowSize;

	for (int j = 0; j < FrequencyBandCount; ++j)
	{
		tab[j] = 0;
	}

	for (int i = 0; i < IterationCount; ++i)
	{
		int SampleStart = i * FFTWindowSize;
		int SampleEnd = (i + 1) * FFTWindowSize - 1;

		for (int j = SampleStart; j <= SampleEnd; ++j)
		{
			SamplesTab[j - SampleStart] = (float)Sound.Samples[j] / 32768.0f;
		}


		kiss_fftr(cfg, SamplesTab, out);

		for (int j = 0; j < FrequencyBandCount; ++j)
		{
			int Start = j*(OutCount / FrequencyBandCount);
			int End = (j + 1)*(OutCount / FrequencyBandCount) - 1;

			for (int k = Start; k <= End; ++k)
			{
				tab[j] += sqrtf(out[k].r*out[k].r + out[k].i*out[k].i);
			}
		}
	}


	double max = 0;

	for (int i = 0; i < FrequencyBandCount; ++i)
	{
		if (tab[i] > max)
		{
			max = tab[i];
		}
	}

	for (int i = 0; i < FrequencyBandCount; ++i)
	{
		//tab[i] = tab[i] / max;
	}

	free(Sound.Free);
	//printf("Przetworzono %s\n", FileName);
}


float Sigmoid(float x)
{
	float Result = 1 / (1 + pow(E, -x));

	return Result;
}

float SigmoidDerivative(float x)
{
	float Result = (x)*(1 - x);

	return Result;
}


neural_network SetupNeuralNetwork(int InputNeuronCount, int OutputNeuronCount, int HiddenLayerCount, int *HiddenLayerNeuronCounts)
{
	neural_network Result = {};

	Result.InputNeuronCount = InputNeuronCount;
	Result.InputNeurons = (float *)malloc(InputNeuronCount*sizeof(float));

	Result.HiddenLayerCount = HiddenLayerCount;
	Result.HiddenLayers = (neuron_layer *)malloc(HiddenLayerCount*sizeof(neuron_layer));

	for (int HiddenLayerIndex = 0;
		HiddenLayerIndex < HiddenLayerCount;
		++HiddenLayerIndex)
	{
		neuron_layer *HiddenLayer = Result.HiddenLayers + HiddenLayerIndex;

		int NeuronCount = HiddenLayerNeuronCounts[HiddenLayerIndex];

		HiddenLayer->NeuronCount = NeuronCount;
		int WeightCount;

		HiddenLayer->Neurons = (float *)malloc(NeuronCount * sizeof(float));

		if (HiddenLayerIndex == 0)
		{
			WeightCount = InputNeuronCount;
		}
		else
		{
			WeightCount = HiddenLayerNeuronCounts[HiddenLayerIndex - 1];
		}

		HiddenLayer->Weights = (float **)malloc(NeuronCount*sizeof(float *));
		HiddenLayer->Biases = (float *)malloc(NeuronCount*sizeof(float));

		for (int NeuronIndex = 0;
			NeuronIndex < NeuronCount;
			++NeuronIndex)
		{
			HiddenLayer->Biases[NeuronIndex] = ((float)(rand() % 100) / 100.0f) - 1;
			HiddenLayer->Weights[NeuronIndex] = (float *)malloc(WeightCount*sizeof(float));
			float *HiddenNeuronWeights = HiddenLayer->Weights[NeuronIndex];
			for (int WeightIndex = 0;
				WeightIndex < WeightCount;
				++WeightIndex)
			{
				HiddenNeuronWeights[WeightIndex] = ((float)(rand() % 200) / 100.0f) - 1;
			}
		}
	}

	Result.OutputNeuronCount = OutputNeuronCount;

	int NeuronCount = OutputNeuronCount;
	int WeightCount;
	if (HiddenLayerCount == 0)
	{
		WeightCount = InputNeuronCount;
	}
	else
	{
		WeightCount = HiddenLayerNeuronCounts[HiddenLayerCount - 1];
	}

	Result.OutputNeurons.Neurons = (float *)malloc(NeuronCount * sizeof(float));
	Result.OutputNeurons.NeuronCount = NeuronCount;

	Result.OutputNeurons.Weights = (float **)malloc(NeuronCount * sizeof(float *));
	Result.OutputNeurons.Biases = (float *)malloc(NeuronCount*sizeof(float));

	for (int NeuronIndex = 0;
		NeuronIndex < NeuronCount;
		++NeuronIndex)
	{
		Result.OutputNeurons.Biases[NeuronIndex] = ((float)(rand() % 100) / 100.0f) - 1;
		Result.OutputNeurons.Weights[NeuronIndex] = (float *)malloc(WeightCount*sizeof(float));
		float *Weights = Result.OutputNeurons.Weights[NeuronIndex];
		for (int WeightIndex = 0;
			WeightIndex < WeightCount;
			++WeightIndex)
		{
			Weights[WeightIndex] = ((float)(rand() % 200) / 100.0f) - 1;
		}
	}

	return Result;
}
