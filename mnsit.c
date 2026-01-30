#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Sigmoid Aktivasyon Fonksiyonu ve Türevi
float sigmoid(float x) {
return 1.0f / (1.0f + expf(-x));
}
float sigmoid_turev(float x) {
return x*(1-x);
}

// Resimler ve Etiketlerin Matrisleri
float resimler[60000][784];
int etiketler[60000];
float resimlerTest[10000][784];
int etiketlerTest[10000];

int main () { 

    // Dosyaların Belirlenmesi
    FILE *goruntuDosyasi;
    FILE *etiketDosyasi;
    FILE *goruntuTestDosyasi;
    FILE *etiketTestDosyasi;
    goruntuDosyasi = fopen("train-images.idx3-ubyte", "rb");
    etiketDosyasi = fopen("train-labels.idx1-ubyte", "rb");
    goruntuTestDosyasi = fopen("t10k-images.idx3-ubyte", "rb");
    etiketTestDosyasi = fopen("t10k-labels.idx1-ubyte", "rb");
    
    // Sahte Okuma
    unsigned char sahteOkuma[16];
    unsigned char sahteOkuma2[8];
    unsigned char sahteOkuma3[16];
    unsigned char sahteOkuma4[8];
    fread(sahteOkuma, 1,16, goruntuDosyasi);
    fread(sahteOkuma2, 1, 8, etiketDosyasi);
    fread(sahteOkuma3, 1,16, goruntuTestDosyasi);
    fread(sahteOkuma4, 1, 8, etiketTestDosyasi);
    
    // Dosyaların Okunup Değişken İçine Atanması
    for(int a=0; a<60000; a=a+1) {
        unsigned char geciciDizi[784];
        fread(geciciDizi,1,784,goruntuDosyasi);

        for (int b=0; b<784; b=b+1) {
        resimler[a][b] = (float)geciciDizi[b]/255.0;
        } 
    }
    for(int a=0; a<60000; a=a+1) {
        unsigned char geciciDizi;
        fread(&geciciDizi,1,1,etiketDosyasi);

        etiketler[a] = (int)geciciDizi;
    }
    for(int a=0; a<10000; a=a+1) {
        unsigned char geciciDizi[784];
        fread(geciciDizi,1,784,goruntuTestDosyasi);

        for (int b=0; b<784; b=b+1) {
        resimlerTest[a][b] = (float)geciciDizi[b]/255.0;
        } 
    }
    for(int a=0; a<10000; a=a+1) {
        unsigned char geciciDizi;
        fread(&geciciDizi,1,1,etiketTestDosyasi);

        etiketlerTest[a] = (int)geciciDizi;
    }

    // Ağırlık ve Bias Matrislerinin OLuşturulması
    float agirliklar1[128][784];
    float agirliklar2[10][128];
    float bias1[128];
    float bias2[10];

    // Ağırlık Matrislerinin Doldurulması
    for (int a=0; a<128; a=a+1) {
        for(int b=0; b<784; b=b+1) {
            agirliklar1[a][b] = (float)rand() / (float)RAND_MAX -0.5;
        }
    }
    for (int a=0; a<10; a=a+1) {
        for(int b=0; b<128; b=b+1) {
            agirliklar2[a][b] = (float)rand() / (float)RAND_MAX -0.5;
        }
    }

    // Bias Matrislerinin Doldurulması
    for (int a=0; a<128; a=a+1) {
            bias1[a] = (float)rand() / (float)RAND_MAX -0.5;
    }
    for (int a=0; a<10; a=a+1) {
            bias2[a] = (float)rand() / (float)RAND_MAX -0.5;
    }

    // Yapay Nöronların Oluşturulması
    /* float girdi[784]; Bu dizi ileride resim okuma eklendiğinde çalışacaktır. */
    float birinciKatman[128];
    float ikinciKatman[10];

    // Öğrenme Algoritması
    #define epoch 20
    #define learning_rate 0.01
    for (int ep=0; ep<epoch; ep=ep+1) {
        for (int a=0; a<60000; a=a+1) {
            for (int c=0; c<128; c=c+1) {
                float toplam = 0;
                for(int b=0; b<784; b=b+1) {
                    toplam = toplam + (resimler[a][b]*agirliklar1[c][b]);
                    }
                toplam = toplam + bias1[c];
                birinciKatman[c] = sigmoid(toplam);
            }  
            for (int d=0; d<10; d=d+1) {
                float toplam = 0;
                for(int e=0; e<128; e=e+1) {
                    toplam = toplam + (birinciKatman[e]*agirliklar2[d][e]);
                    }
                toplam = toplam + bias2[d];
                ikinciKatman[d] = sigmoid(toplam);
            }
            int hedef[10] = {0,0,0,0,0,0,0,0,0,0};
            hedef[etiketler[a]] = 1;
            float hata1[10];
            float hata2[128];
            for (int g=0; g<10; g=g+1) {
                hata1[g] = hedef[g]-ikinciKatman[g];
                hata1[g] = hata1[g]*sigmoid_turev(ikinciKatman[g]);
            }
            for (int g=0; g<128; g=g+1) { 
                float toplam = 0;
                for (int h=0; h<10; h=h+1) {
                    toplam = toplam + hata1[h]*agirliklar2[h][g];
                }
                hata2[g] = toplam * sigmoid_turev(birinciKatman[g]);
            }
            for (int i=0; i<128; i=i+1) {
                for (int j=0; j<784; j=j+1) {
                    agirliklar1[i][j] = agirliklar1[i][j] + (learning_rate *hata2[i] * resimler[a][j]);
                }
                bias1[i] = bias1[i] + learning_rate * hata2[i];
            }
            for (int i=0; i<10; i=i+1) {
                for (int j=0; j<128; j=j+1) {
                    agirliklar2[i][j] = agirliklar2[i][j] + (learning_rate * hata1[i] * birinciKatman[j]);
                }
                bias2[i] = bias2[i] + learning_rate * hata1[i];
            }
            if (a % 1000 == 0) {
            printf("Epoch: %d, Resim: %d tamamlandi.\n", ep+1, a);
            }
        }
    }

    // İleri Yayılım Algoritması
    int dogruTahmin = 0;
    for (int a = 0; a < 10000; a++) {
    for (int c = 0; c < 128; c++) {
        float toplam = 0;
        for (int b = 0; b < 784; b++) {
            toplam += resimlerTest[a][b] * agirliklar1[c][b];
        }
        birinciKatman[c] = sigmoid(toplam + bias1[c]);
    }
    for (int d = 0; d < 10; d++) {
        float toplam = 0;
        for (int e = 0; e < 128; e++) {
            toplam += birinciKatman[e] * agirliklar2[d][e];
        }
        ikinciKatman[d] = sigmoid(toplam + bias2[d]);
    }

    // En Yüksek Değeri Bulma
    int tahmin = 0;
    float maksimumDeger = ikinciKatman[0];
    for (int i = 1; i < 10; i++) {
        if (ikinciKatman[i] > maksimumDeger) {
            maksimumDeger = ikinciKatman[i];
            tahmin = i;
        }
    }
    if (tahmin == etiketlerTest[a]) {
        dogruTahmin++;
    }
    }
    printf("Test Basarisi: %.2f\n", (float)dogruTahmin / 100.0);
}
