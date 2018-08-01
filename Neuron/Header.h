#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <windows.h>
#include <ctime>

#include "kiss_fft.h"
#include "kiss_fftr.h"

#define FFT_WINDOW_SIZE 4096
#define FREQUENCY_BAND_COUNT 500

#define CAT 0
#define DOG 1

#define E 2.718281828459f

#define ArrayCount(Array) (sizeof(Array)/sizeof((Array)[0]))

#pragma pack(push, 1)

struct WAVE_header
{
	unsigned int RIFFID;
	unsigned int Size;
	unsigned int WAVEID;
};

#define RIFF_CODE(a, b, c, d) ((unsigned int)(a << 0) | (unsigned int)(b << 8) | (unsigned int)(c << 16) | (unsigned int)(d << 24))

enum
{
	WAVE_ChunkID_fmt = RIFF_CODE('f', 'm', 't', ' '),
	WAVE_ChunkID_data = RIFF_CODE('d', 'a', 't', 'a'),
	WAVE_ChunkID_RIFF = RIFF_CODE('R', 'I', 'F', 'F'),
	WAVE_ChunkID_WAVE = RIFF_CODE('W', 'A', 'V', 'E'),
};

struct WAVE_chunk
{
	unsigned int ID;
	unsigned int Size;
};

struct WAVE_fmt
{
	unsigned short wFormatTag;
	unsigned short nChannels;
	unsigned int nSamplesPerSec;
	unsigned int nAvgBytesPerSec;
	unsigned short nBlockAlign;
	unsigned short wBitsPerSample;
	unsigned short cbSize;
	unsigned short wValidBitsPerSample;
	unsigned int dwChannelMask;
	unsigned char SubFormat[16];
};

#pragma pack(pop)

struct entire_file
{
	unsigned int ContentsSize;
	void *Contents;
};

struct riff_iterator
{
	unsigned char *At;
	unsigned char *Stop;
};

struct loaded_sound
{
	unsigned int SampleCount; // NOTE: This is SampleCount divided by 8
	unsigned int ChannelCount;
	unsigned int SamplesPerSec;
	float Time;
	short *Samples;
	void *Free;
};

struct neuron_layer
{
	int NeuronCount;
	float *Neurons;
	float **Weights;
	float *Biases;
};

struct neural_network
{
	int InputNeuronCount;
	int HiddenLayerCount;
	int OutputNeuronCount;
	float *InputNeurons;
	neuron_layer *HiddenLayers;
	neuron_layer OutputNeurons;
};

loaded_sound LoadWAV(char *FileName);
void FourierTransform(char *FileName, kiss_fftr_cfg &cfg, kiss_fft_cpx *out, float *tab, int OutCount, float *SamplesTab, int FFTWindowSize, int FrequencyBandCount);
neural_network SetupNeuralNetwork(int InputNeuronCount, int OutputNeuronCount, int HiddenLayerCount, int *HiddenLayerNeuronCounts);
float SigmoidDerivative(float x);
float Sigmoid(float x);