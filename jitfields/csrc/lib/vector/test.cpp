#include "vector.h"
#include "stream.h"
#include <iostream>
#include <tuple>

using namespace jf;
using namespace std;


int main() {
    const int * ptr = reinterpret_cast<const int *>(1L);
    Vector<int, 5> svec(1);
    Vector<int> dvec(5, 1);
    Vector<int, 3> a = {1, 2, 3};

    int b, c, d;
    a.unbind(b, d, d);

    cout << weak_ref(ptr)       << endl << endl;
    cout << weak_ref(ptr, 2)    << endl << endl;
    cout << weak_ref(5, ptr)    << endl << endl;
    cout << weak_ref(5, ptr, 2) << endl << endl;
    cout << svec << endl << endl;
    cout << dvec << endl << endl;
}
