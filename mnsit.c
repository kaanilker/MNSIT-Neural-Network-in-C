#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Sigmoid ve ReLU Aktivasyon Fonksiyonu ve Türevi
float sigmoid(float x) {
return 1.0f / (1.0f + expf(-x));
}
float sigmoidTurev(float x) {
return x*(1-x);
}

float relu(float x) {
    if (x<=0) {
        return 0;
    } else {
        return x;
    }
}
float reluTurev(float x) {
    if (x<=0) {
        return 0;
    } else {
        return 1;
    }
}

// Resimler ve Etiketlerin Matrisleri
float resimler[60000][784];
int etiketler[60000];
float resimlerTest[10000][784];
int etiketlerTest[10000];

int main () { 

    // Model Hiperparametreleri
    #define epoch 10
    #define learningRate 0.1
    #define batchSize 64
    #define hiddenLayer 256

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
    fread(sahteOkuma, 1,16, goruntuDosyasi);
    fread(sahteOkuma, 1, 8, etiketDosyasi);
    fread(sahteOkuma, 1,16, goruntuTestDosyasi);
    fread(sahteOkuma, 1, 8, etiketTestDosyasi);
    
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
    float agirliklar1[hiddenLayer][784];
    float agirliklar2[10][hiddenLayer];
    float bias1[hiddenLayer];
    float bias2[10];

    // Ağırlık Matrislerinin Doldurulması
    for (int a=0; a<hiddenLayer; a=a+1) {
        for(int b=0; b<784; b=b+1) {
            float std = sqrtf(2.0f / 784.0f);
            agirliklar1[a][b] = ((float)rand() / RAND_MAX - 0.5) * 2 * std;
        }
    }
    for (int a=0; a<10; a=a+1) {
        for(int b=0; b<hiddenLayer; b=b+1) {
            float std = sqrtf(2.0f / (float)hiddenLayer);
            agirliklar2[a][b] = ((float)rand() / RAND_MAX - 0.5) * 2 * std;
        }
    }

    // Bias Matrislerinin Doldurulması
    for (int a=0; a<hiddenLayer; a=a+1) {
            bias1[a] = ((float)rand() / RAND_MAX - 0.5) * 0.01;
    }
    for (int a=0; a<10; a=a+1) {
            bias2[a] = ((float)rand() / RAND_MAX - 0.5) * 0.01;
    }

    // Yapay Nöronların Oluşturulması
    /* float girdi[784]; Bu dizi ileride resim okuma eklendiğinde çalışacaktır. */
    float birinciKatman[hiddenLayer];
    float ikinciKatman[10];

    // Öğrenme Algoritması
    for (int a=0; a<epoch; a+=1) {
        for (int b=0; b<60000; b+=1) {

            // İleri Yayılım
            for (int c=0; c<hiddenLayer; c+=1) {
                float toplam = 0;
                for(int d=0; d<784; d+=1) {
                    toplam += (resimler[b][d]*agirliklar1[c][d]);
                    }
                toplam += bias1[c];
                birinciKatman[c] = relu(toplam);
            }  
            for (int c=0; c<10; c+=1) {
                float toplam = 0;
                for(int d=0; d<hiddenLayer; d+=1) {
                    toplam += (birinciKatman[d]*agirliklar2[c][d]);
                    }
                toplam += bias2[c];
                ikinciKatman[c] = sigmoid(toplam);
            }

            // Hata Algoritması ve Hedef
            int hedef[10] = {0,0,0,0,0,0,0,0,0,0};
            hedef[etiketler[b]] = 1;
            float hata1[10];
            float hata2[hiddenLayer];

            // Geri Yayılım Algoritması
            for (int c=0; c<10; c+=1) {
                hata1[c] = hedef[c]-ikinciKatman[c];
                hata1[c] = hata1[c]*sigmoidTurev(ikinciKatman[c]);
            }
            for (int c=0; c<hiddenLayer; c+=1) { 
                float toplam = 0;
                for (int d=0; d<10; d+=1) {
                    toplam += hata1[d]*agirliklar2[d][c];
                }
                hata2[c] = toplam * reluTurev(birinciKatman[c]);
            }
            for (int c=0; c<hiddenLayer; c+=1) {
                for (int d=0; d<784; d+=1) {
                    agirliklar1[c][d] += (learningRate *hata2[c] * resimler[b][d]);
                }
                bias1[c] += learningRate * hata2[c];
            }
            for (int c=0; c<10; c+=1) {
                for (int d=0; d<hiddenLayer; d+=1) {
                    agirliklar2[c][d] += (learningRate * hata1[c] * birinciKatman[d]);
                }
                bias2[c] += learningRate * hata1[c];
            }
            if (b % 500 == 0) {
            printf("Epoch: %d, Resim: %d tamamlandi.\n", a+1, b);
            }
        }
    }

    // Test Algoritması
    int dogruTahmin = 0;
    for (int a=0; a<10000; a+=1) {
        for (int b=0; b<hiddenLayer; b+=1) {
            float toplam = 0;
            for (int c=0; c<784; c+=1) {
                toplam += resimlerTest[a][c] * agirliklar1[b][c];
            }
            birinciKatman[b] = relu(toplam + bias1[b]);
        }
        for (int b=0; b<10; b+=1) {
            float toplam = 0;
            for (int c=0; c<hiddenLayer; c+=1) {
                toplam += birinciKatman[c] * agirliklar2[b][c];
            }
            ikinciKatman[b] = sigmoid(toplam + bias2[b]);
        }

        // En Yüksek Değeri Bulma
        int tahmin = 0;
        float maksimumDeger = ikinciKatman[0];
        for (int b=1; b<10; b+=1) {
            if (ikinciKatman[b] > maksimumDeger) {
                maksimumDeger = ikinciKatman[b];
                tahmin = b;
            }
        }

        // SOn 10 Tahmin
        if (a >= 9990) {
        printf("Resim %d: Tahmin=%d, Gercek=%d, Cikislar=[", 
               a, tahmin, etiketlerTest[a]);
        for (int b=0; b<10; b+=1) {
            printf("%.3f", ikinciKatman[b]);
            if (b < 9) printf(", ");
        }
        printf("], Dogru=%s\n", 
               (tahmin == etiketlerTest[a]) ? "EVET" : "HAYIR");
        }
        if (tahmin == etiketlerTest[a]) {
            dogruTahmin+=1;
        }
    }
    printf("Test Basarisi: %%%.2f\n", (float)dogruTahmin / 10000.0 * 100.0);
}
