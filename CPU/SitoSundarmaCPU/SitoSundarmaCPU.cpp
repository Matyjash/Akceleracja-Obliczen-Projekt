
#include <iostream>

void sieveOfSundaram(int n) {
    
    int k = (n - 2) / 2;
    bool* numberArray = new bool[k + 1];
    for (int i = 0; i < k + 1; i++)
    {
        numberArray[i] = true;
    }

    for (int i = 1; i < k + 1; i++) 
    {
        int j = i;
        while (i + j + 2 * i * j <= k) 
        {
            numberArray[i + j + 2 * i * j] = false;
            j++;
        }
    }
    if (n > 2)
    {
        std::cout << 2 << " ";
    }
    for (int i = 1; i < k + 1; i++)
    {
        if (numberArray[i])
        {
            std::cout << 2 * i + 1 << " ";
        }
    }
    delete[] numberArray;
    return;
}

int main()
{
    std::cout << "Hello World!\n";
    sieveOfSundaram(31);
}
