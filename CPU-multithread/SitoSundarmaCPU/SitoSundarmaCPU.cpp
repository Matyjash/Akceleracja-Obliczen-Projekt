
#include <iostream>
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
    if (n > 2)
    {
        std::cout << 2 << " ";
    }
    for (int i = 1; i < k + 1; i++)
    {
        if (numberArray[i])
        {
            //liczba w postaci 2k + 1
           // std::cout << 2 * i + 1 << " ";
        }
    }
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
    //wypisujemy dwójkę jako najmniejszą liczbę pierwszą
    for (int i = startingIndex; i < endingIndex; i++)
    {
        if (numberArray[i])
        {
            //liczba w postaci 2k + 1
           // std::cout << 2 * i + 1;
        }
    }    
    return;
}

void sieveOfSundaramMultiThreaded(int n) {

    int k = (n - 2) / 2;
    bool* numberArray = new bool[k + 1];
    int divider1 = ceil(float(k) / 4);
    int divider2 = 2 * ceil(float(k) / 4);
    int divider3 = 3 * ceil(float(k) / 4);
    
    for (int i = 0; i < k+1; i++)
        numberArray[i] = true;


    std::thread t1(sundaramThread, numberArray, 1, divider1, k);
    std::thread t2(sundaramThread, numberArray, divider1, divider2, k);
    std::thread t3(sundaramThread, numberArray, divider2, divider3, k);
    std::thread t4(sundaramThread, numberArray, divider3, k+1, k);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    if (k > 2)
    {
        //std::cout << 2 << "\n";
    }
    delete[] numberArray;
    return;
}

int main()
{
    //dla pomiarów czasowych
    long long int frequency, start, elapsed;
    QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);

    std::cout << "Podaj n:";
    int n;
    std::cin >> n;
    std::cout << "\n";

    start = read_QPC();
    sieveOfSundaram(n);
    elapsed = read_QPC() - start;
    std::cout << "\nCzas jeden watek[ms] = " << (1000.0 * elapsed) / frequency << "\n";


    start = read_QPC();
    sieveOfSundaramMultiThreaded(n);
    elapsed = read_QPC() - start;
    std::cout << "\nCzas wiele watkow[ms] = " << (1000.0 * elapsed) / frequency << "\n";

}
