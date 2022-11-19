
#include <iostream>
#include <vector>
#include<thread>
#include <windows.h>
#include <iomanip>

long long int read_QPC()
{
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return((long long int)count.QuadPart);
}

void sieveOfSundaram(int n) {
    
    int k = (n - 2) / 2;
    bool* numberArray = new bool[k + 1];
    //inicjalizujemy tablice nadając jej wszystkim elementom wartości true (na początku uznajemy każdą liczbę za pierwszą)
    for (int i = 0; i < k + 1; i++)
        numberArray[i] = true;

    for (__int64 i = 1; i < k + 1; i++) 
    {
        __int64 j = i;
        while (i + j + 2 * i * j <= k) 
        {
            //jeżeli mamy liczbę nieparzystą w postaci 2k+1 (uzyskujemy ją przy wypisywaniu wyniku) 
            //to jeżeli liczba ta ma postać i+j+2ij to możemy ją wykluczyć ze zbioru liczb nieparzystych
            numberArray[i + j + 2 * i * j] = false;
            j++;
        }
    }
    //wypisujemy dwójkę jako najmniejszą liczbę pierwszą
    /*
    if (n > 2)
    {
        std::cout << 2 << " ";
    }

    for (int i = 1; i < k + 1; i++)
    {
        if (numberArray[i])
        {
            //liczba w postaci 2k + 1
           //std::cout << 2 * i + 1 << " ";
        }
    }
     */
    delete[] numberArray;
    return;
}


void sundaramThread(bool* numberArray, int startingIndex, int endingIndex, int k) {
    

    for (__int64 i = startingIndex; i < endingIndex; i++)
    {
        __int64 j = i;
        while (i + j + 2 * i * j <= k)
        {
            //jeżeli mamy liczbę nieparzystą w postaci 2k+1 (uzyskujemy ją przy wypisywaniu wyniku) 
            //to jeżeli liczba ta ma postać i+j+2ij to możemy ją wykluczyć ze zbioru liczb nieparzystych
            numberArray[i + j + 2 * i * j] = false;
            j++;
        }
    }
    /*

    for (int i = 1; i < k + 1; i++)
    {
        if (numberArray[i])
        {
            //liczba w postaci 2k + 1
           //std::cout << 2 * i + 1 << " ";
        }
    }
     */
    return;
}

void sieveOfSundaramMultiThreaded(int n, int threadCount) {

    int k = (n - 2) / 2;
    bool* numberArray = new bool[k + 1];
    int divider1 = ceil(float(k) / 4);
    int divider2 = 2 * ceil(float(k) / 4);
    int divider3 = 3 * ceil(float(k) / 4);
    
    for (int i = 0; i < k+1; i++)
        numberArray[i] = true;

    std::vector<std::thread> threadPool;

    //wypisujemy dwójkę jako najmniejszą liczbę pierwszą
    /*
    if (n > 2)
    {
        std::cout << 2 << " ";
    }
    */
    for (int i = 0; i < threadCount; i++) {
        int divider = i * ceil(float(k) / threadCount);
        int divider2 = i + 1 * ceil(float(k) / threadCount);
        if (i == 0) {
            std::thread t1(sundaramThread, numberArray, 1, divider1, k);
            threadPool.push_back(std::move(t1));
        }
        else if(i==threadCount-1) {
            std::thread t1(sundaramThread, numberArray, divider, k+1, k);
            threadPool.push_back(std::move(t1));
        }
        else {
            std::thread t1(sundaramThread, numberArray, divider, divider1, k);
            threadPool.push_back(std::move(t1));
        }

    }
    for (int i = 0; i < threadCount; i++) {
        threadPool[i].join();
    }
    delete[] numberArray;
    return;
}

int main()
{
    //dla pomiarów czasowych
    long long int frequency, start, elapsed;
    QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);

    std::cout << "Podaj n: ";
    int n;
    std::cin >> n;
    std::cout << "\n";

    std::cout << "Podaj liczbe watkow: ";
    int threads;
    std::cin >> threads;
    std::cout << "\n";

    start = read_QPC();
    sieveOfSundaram(n);
    elapsed = read_QPC() - start;
    std::cout << "\nCzas jeden watek[ms] = " << (1000.0 * elapsed) / frequency << "\n";


    start = read_QPC();
    sieveOfSundaramMultiThreaded(n, threads);
    elapsed = read_QPC() - start;
    std::cout << "\nCzas wiele watkow[ms] = " << (1000.0 * elapsed) / frequency << "\n";

}
