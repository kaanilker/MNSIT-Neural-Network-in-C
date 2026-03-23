#ifndef REQUIREMENTS_H
#define REQUIREMENTS_H

static inline float sigmoid(float x)      { return 1.0f / (1.0f + expf(-x)); }
static inline float sigmoidTurev(float x) { return x * (1 - x); }
static inline float relu(float x)         { return x <= 0 ? 0 : x; }
static inline float reluTurev(float x)    { return x <= 0 ? 0 : 1; }
void diziSifirlama(float *dizi, int boyut);
void matrisSifirlama(float **matris, int satir, int sutun);
void goruntuOkuma(FILE *dosya, float matris[][784], int adet);
void etiketOkuma(FILE *dosya, int *dizi, int adet);
void agirlikDoldurma(float *matris, int satir, int sutun);
void biasDoldurma(float *bias, int boyut);

#endif
