#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "requirements.h"

// Sıfırlama Fonksiyonları
void diziSifirlama(float *dizi, int boyut) {
    for (int x = 0; x < boyut; x+=1)
        dizi[x] = 0.0f;
}
void matrisSifirlama(float **matris, int satir, int sutun) {
    for (int x = 0; x < satir; x+=1)
        for (int y = 0; y < sutun; y+=1)
            matris[x][y] = 0.0f;
}

// Doldurma Fonksiyonları
void agirlikDoldurma(float *matris, int satir, int sutun) {
    float std = sqrtf(2.0f / sutun);
    for (int x = 0; x < satir * sutun; x+=1)
        matris[x] = ((float)rand() / RAND_MAX - 0.5f) * 2 * std;
}
void biasDoldurma(float *bias, int boyut) {
    for (int x = 0; x < boyut; x+=1)
        bias[x] = 0.0f;
}

// Dosya Okuma Fonksiyonları
void goruntuOkuma(FILE *dosya, float matris[][784], int adet) {
    for (int a = 0; a < adet; a+=1) {
        unsigned char geciciDizi[784];
        fread(geciciDizi, 1, 784, dosya);
        for (int b = 0; b < 784; b+=1)
            matris[a][b] = (float)geciciDizi[b] / 255.0f;
    }
}
void etiketOkuma(FILE *dosya, int *dizi, int adet) {
    for (int a = 0; a < adet; a+=1) {
        unsigned char gecici;
        fread(&gecici, 1, 1, dosya);
        dizi[a] = (int)gecici;
    }
}
