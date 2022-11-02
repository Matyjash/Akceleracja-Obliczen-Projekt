
#include <iostream>

void sieveOfSundaram(int n) {
    
    int k = (n - 2) / 2;
    bool* numberArray = new bool[k + 1];
    //inicjalizujemy tablice nadając jej wszystkim elementom wartości true (na początku uznajemy każdą liczbę za pierwszą)
    for (int i = 0; i < k + 1; i++)
        numberArray[i] = true;

    for (int i = 1; i < k + 1; i++) 
    {
        int j = i;
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
            std::cout << 2 * i + 1 << " ";
        }
    }
    delete[] numberArray;
    return;
}

int main()
{
    std::cout << "Podaj n:";
    int n;
    std::cin >> n;
    std::cout << "\n";
    sieveOfSundaram(n);
}
